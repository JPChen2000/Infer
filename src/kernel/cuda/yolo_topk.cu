// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

// Include CUB before Tensor's legacy BYTE_LEN macro, which collides with a
// template parameter in CUDA 11.8's cub/util_ptx.cuh.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <climits>
#include <memory>
#include <vector>

#include "src/kernel/yolo_selection.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

// Temporary allocations belong to the active execution context and return to
// its pool. All uses and subsequent reuses run on that context's CUDA stream.
class TopKWorkspace {
 public:
    ~TopKWorkspace() { cuda_detail::ReleaseTemporaryDeviceBuffer(data_, bytes_); }
    bool Allocate(size_t bytes) {
        bytes_ = bytes;
        return cuda_detail::AcquireTemporaryDeviceBuffer(bytes, &data_) == 0;
    }
    template <typename T> T* As() const { return static_cast<T*>(data_); }
 private:
    void* data_{nullptr};
    size_t bytes_{0};
};

struct SegmentOffset {
    int length;
    __host__ __device__ int operator()(int segment) const { return segment * length; }
};

using SegmentOffsets = cub::TransformInputIterator<int, SegmentOffset, cub::CountingInputIterator<int>>;

__global__ void PrepareTopKKeys(const uint32_t* data, uint32_t* keys, int64_t* indices,
                               int64_t count, int64_t length, int64_t inner) {
    const int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= count) return;
    const int64_t segment = position / length;
    const int64_t index = position % length;
    const int64_t source = (segment / inner * length + index) * inner + segment % inner;
    const uint32_t bits = data[source];
    // Monotonic float keys, with all NaNs greatest and both zero signs equal.
    // Stable radix sorting consequently resolves every tie by original index.
    const uint32_t magnitude = bits & 0x7fffffffu;
    keys[position] = magnitude > 0x7f800000u ? 0xffffffffu :
                     magnitude == 0 ? 0x80000000u :
                     (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
    indices[position] = index;
}

__global__ void WriteTopK(const uint32_t* input, const int64_t* order, uint32_t* output,
                         int64_t* output_indices, int64_t count, int64_t length,
                         int64_t inner, int64_t k) {
    const int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= count) return;
    const int64_t lane = position % inner;
    const int64_t selected = position / inner % k;
    const int64_t batch = position / inner / k;
    const int64_t index = order[(batch * inner + lane) * length + selected];
    const int64_t source = (batch * length + index) * inner + lane;
    // Copy storage bits to preserve signed zeros and NaN payloads exactly.
    output[position] = input[source];
    output_indices[position] = index;
}

cudaError_t SortTopK(void* temporary, size_t& bytes, uint32_t* keys_in,
                    uint32_t* keys_out, int64_t* indices_in, int64_t* indices_out,
                    int count, int segments, int length, bool largest) {
    const auto stream = cuda_detail::InferenceStream();
    if (segments == 1) {
        if (largest) {
            return cub::DeviceRadixSort::SortPairsDescending(temporary, bytes, keys_in, keys_out,
                                                            indices_in, indices_out, count, 0, 32, stream);
        }
        return cub::DeviceRadixSort::SortPairs(temporary, bytes, keys_in, keys_out,
                                             indices_in, indices_out, count, 0, 32, stream);
    }
    SegmentOffsets begin(cub::CountingInputIterator<int>(0), SegmentOffset{length});
    SegmentOffsets end(cub::CountingInputIterator<int>(1), SegmentOffset{length});
    if (largest) {
        return cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temporary, bytes, keys_in, keys_out, indices_in, indices_out, count, segments,
            begin, end, 0, 32, stream);
    }
    return cub::DeviceSegmentedRadixSort::SortPairs(
        temporary, bytes, keys_in, keys_out, indices_in, indices_out, count, segments,
        begin, end, 0, 32, stream);
}

Status TopKShapeSupport(const Tensor* input) {
    if (!input || input->data_type() != DataType::FP32 || input->dims().size() == 0 ||
        input->dims().size() > cuda_detail::kMaxCudaRank) {
        return Status::Error(StatusCode::kUnsupported, "CUDA TopK requires a non-scalar FP32 tensor of rank <= 8");
    }
    size_t elements = 0;
    if (!CheckedElementCount(input->dims().data(), &elements) || elements > INT_MAX) {
        return Status::Error(StatusCode::kUnsupported, "CUDA TopK radix sort requires at most INT_MAX elements");
    }
    return Status::Ok();
}

class CudaYoloTopKKernel final : public KernelBase {
 public:
    Status Supports() const override {
        const auto* param = ParamAs<operators::YoloSelectionParam>();
        if (!param || param->kind != operators::YoloSelectionKind::kTopK) {
            return Status::Error(StatusCode::kUnsupported, "CUDA TopK requires TopK parameters");
        }
        return TopKShapeSupport(param->data.get());
    }

    int32_t compute() override {
        AutoTimer timer("CUDA::TopK::FP32");
        auto* param = ParamAs<operators::YoloSelectionParam>();
        if (!param || !Supports().ok() || !param->control || !param->out || !param->indices_out ||
            !param->control->IsInitialized() || param->control->data_type() != DataType::INT64 ||
            param->control->numel() != 1 || param->control->dims().size() != 1 ||
            param->out->data_type() != DataType::FP32 || param->indices_out->data_type() != DataType::INT64) return -1;
        const auto& dims = param->data->dims().data();
        const int64_t rank = static_cast<int64_t>(dims.size());
        const int64_t axis = param->axis < 0 ? param->axis + rank : param->axis;
        if (axis < 0 || axis >= rank) return -1;
        // Constant K stays on the host. A dynamic K may be produced by a CUDA
        // node, so synchronize just this control before checking its value.
        if (!param->control->is_immutable() && cuda_detail::SyncTensorToHostIfNeeded(
                param->control.get(), sizeof(int64_t), param->control->raw_data()) != 0) return -1;
        const int64_t k = param->control->data<int64_t>()[0];
        const int64_t length = dims[axis];
        if (k < 0 || k > length) return -1;
        auto expected = dims;
        expected[axis] = k;
        if (param->out->dims().data() != expected || param->indices_out->dims().data() != expected) return -1;
        if (param->out->numel() == 0) return 0;
        if (!cuda_detail::IsTensorReady<DataType::FP32>(param->data.get()) ||
            !cuda_detail::IsOutputReady<DataType::FP32>(param->out.get(), &expected) ||
            !cuda_detail::IsOutputReady<DataType::INT64>(param->indices_out.get(), &expected)) return -1;
        const int64_t inner = cuda_detail::ComputeProduct(dims, static_cast<size_t>(axis + 1), dims.size());
        const int count = static_cast<int>(param->data->numel());
        const int segments = static_cast<int>(count / length);
        cuda_detail::DeviceBuffer<float> data;
        cuda_detail::DeviceBuffer<float> output;
        cuda_detail::DeviceBuffer<int64_t> indices;
        if (cuda_detail::CopyTensorToDevice(param->data.get(), &data) != 0 ||
            cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0 ||
            cuda_detail::AllocateTensorOnDevice(param->indices_out.get(), &indices) != 0) return -1;

        TopKWorkspace keys_in, keys_out, order_in, order_out, temporary;
        const size_t key_bytes = static_cast<size_t>(count) * sizeof(uint32_t);
        const size_t order_bytes = static_cast<size_t>(count) * sizeof(int64_t);
        if (!keys_in.Allocate(key_bytes) || !keys_out.Allocate(key_bytes) ||
            !order_in.Allocate(order_bytes) || !order_out.Allocate(order_bytes)) return -1;
        size_t temporary_bytes = 0;
        if (cuda_detail::CudaCheck(SortTopK(nullptr, temporary_bytes, keys_in.As<uint32_t>(),
                keys_out.As<uint32_t>(), order_in.As<int64_t>(), order_out.As<int64_t>(),
                count, segments, static_cast<int>(length), param->largest)) != 0 ||
            !temporary.Allocate(temporary_bytes)) return -1;
        const auto stream = cuda_detail::InferenceStream();
        PrepareTopKKeys<<<static_cast<int>(cuda_detail::DivUp(count, cuda_detail::kCudaThreads)),
                         cuda_detail::kCudaThreads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(data.get()), keys_in.As<uint32_t>(),
            order_in.As<int64_t>(), count, length, inner);
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0 ||
            cuda_detail::CudaCheck(SortTopK(temporary.As<void>(), temporary_bytes, keys_in.As<uint32_t>(),
                keys_out.As<uint32_t>(), order_in.As<int64_t>(), order_out.As<int64_t>(),
                count, segments, static_cast<int>(length), param->largest)) != 0) return -1;
        // sorted=0 permits sorted output as well, so both modes use this path.
        WriteTopK<<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                    cuda_detail::kCudaThreads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(data.get()), order_out.As<int64_t>(),
            reinterpret_cast<uint32_t*>(output.get()), indices.get(), param->out->numel(), length, inner, k);
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
        if (cuda_detail::CopyDeviceToTensor(&output, param->out.get()) != 0) return -1;
        return cuda_detail::CopyDeviceToTensor(&indices, param->indices_out.get());
    }
};

}  // namespace

void EnsureCudaYoloTopKKernelsRegistered() {
    auto& dispatcher = KernelDispatcher::instance();
    static const bool registered = [&dispatcher] {
        dispatcher.RegisterCandidate(
            DeviceType::CUDA, DataType::FP32, DataLayout::ND, "TopK", "cuda_fp32_stable_radix_topk",
            [] { return std::make_unique<CudaYoloTopKKernel>(); },
            [](const KernelRequest& request) {
                // A request without tensors is used when cloning by metadata;
                // Supports() validates its bound parameter snapshot afterward.
                if (request.tensors.empty()) return Status::Ok();
                return TopKShapeSupport(request.tensors[0].get());
            });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
