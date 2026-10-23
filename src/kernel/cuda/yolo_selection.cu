// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include "src/kernel/yolo_selection.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace {
namespace cd = cuda_detail;
using operators::YoloSelectionParam;

Status SelectionShapeSupport(const Tensor* tensor) {
    if (!tensor || tensor->dims().size() > cd::kMaxCudaRank) {
        return Status::Error(StatusCode::kUnsupported, "CUDA selection requires tensor rank <= 8");
    }
    return Status::Ok();
}

Status SelectionRequestSupport(const KernelRequest& request) {
    for (const auto& tensor : request.tensors) {
        // Control and secondary output tensors are optional for some operators.
        // Required tensors are validated by the individual kernel's Supports.
        if (!tensor) continue;
        const auto status = SelectionShapeSupport(tensor.get());
        if (!status.ok()) return status;
    }
    return Status::Ok();
}

// Checked indexing/arithmetic reports malformed model inputs to the caller.
// The flag lives in the active session's temporary pool, never in global state.
class DeviceError {
 public:
    ~DeviceError() { if (flag_) cd::ReleaseTemporaryDeviceBuffer(flag_, sizeof(int)); }
    int Init() {
        void* ptr = nullptr;
        if (cd::AcquireTemporaryDeviceBuffer(sizeof(int), &ptr) != 0) return -1;
        flag_ = static_cast<int*>(ptr);
        return cd::CudaCheck(cudaMemsetAsync(flag_, 0, sizeof(int), cd::InferenceStream()));
    }
    int* get() { return flag_; }
    int Check() {
        int error = 0;
        if (cd::CudaCheck(cudaMemcpyAsync(&error, flag_, sizeof(error), cudaMemcpyDeviceToHost,
                                        cd::InferenceStream())) != 0 ||
            cd::SynchronizeInferenceStream() != 0) return -1;
        return error ? -1 : 0;
    }
 private:
    int* flag_{nullptr};
};

struct ReductionSpec {
    int kept_rank{0};
    int reduced_rank{0};
    int64_t kept_dims[cd::kMaxCudaRank]{};
    int64_t kept_strides[cd::kMaxCudaRank]{};
    int64_t reduced_dims[cd::kMaxCudaRank]{};
    int64_t reduced_strides[cd::kMaxCudaRank]{};
    int64_t reduction_count{1};
};

// Choose the first source element on ties, including NaNs and signed zero.
// Moving raw bits preserves the selected NaN payload just like Common.
__device__ bool BetterMax(uint32_t bits, int64_t index, uint32_t old_bits, int64_t old_index) {
    if (index < 0) return false;
    if (old_index < 0) return true;
    const float value = __uint_as_float(bits), old = __uint_as_float(old_bits);
    const bool nan = isnan(value), old_nan = isnan(old);
    if (nan != old_nan) return nan;
    if (!nan && value != old) return value > old;
    return index < old_index;
}

__global__ void ReduceMaxCuda(const uint32_t* input, uint32_t* output, int64_t count, ReductionSpec spec) {
    const int lane = threadIdx.x % 32;
    const int64_t first_row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    for (int64_t row = first_row; row < count; row += static_cast<int64_t>(gridDim.x) * (blockDim.x / 32)) {
    int64_t remaining = row, base = 0;
    for (int i = spec.kept_rank - 1; i >= 0; --i) {
        base += (remaining % spec.kept_dims[i]) * spec.kept_strides[i];
        remaining /= spec.kept_dims[i];
    }
    uint32_t best_bits = 0xff800000u;
    int64_t best_index = -1;
    for (int64_t r = lane; r < spec.reduction_count; r += 32) {
        int64_t coordinate = r, source = base;
        for (int i = spec.reduced_rank - 1; i >= 0; --i) {
            source += (coordinate % spec.reduced_dims[i]) * spec.reduced_strides[i];
            coordinate /= spec.reduced_dims[i];
        }
        const uint32_t bits = input[source];
        if (BetterMax(bits, source, best_bits, best_index)) { best_bits = bits; best_index = source; }
    }
    for (int distance = 16; distance; distance /= 2) {
        const uint32_t bits = __shfl_down_sync(0xffffffffu, best_bits, distance);
        const int64_t index = __shfl_down_sync(0xffffffffu, static_cast<long long>(best_index), distance);
        if (lane + distance < 32 && BetterMax(bits, index, best_bits, best_index)) {
            best_bits = bits; best_index = index;
        }
    }
    if (lane == 0) output[row] = best_bits;
    }
}

int RunReduceMax(YoloSelectionParam* param) {
    if (!param || !param->data || !param->out || param->data->data_type() != DataType::FP32 ||
        param->out->data_type() != DataType::FP32) return -1;
    if (param->control && param->control->numel() > 0 && !param->control->is_immutable()) {
        size_t bytes = 0;
        if (!cd::SafeByteCount(param->control->numel(), sizeof(int64_t), &bytes) ||
            cd::SyncTensorToHostIfNeeded(param->control.get(), bytes, param->control->raw_data()) != 0) return -1;
    }
    std::vector<int64_t> axes;
    if (!operators::YoloSelectionAxes(*param, &axes)) return -1;
    const auto& dims = param->data->dims().data();
    if (dims.size() > cd::kMaxCudaRank) return -1;
    std::vector<int64_t> expected;
    ReductionSpec spec;
    int64_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        if (dims[i] < 0 || (dims[i] && stride > std::numeric_limits<int64_t>::max() / dims[i])) return -1;
        stride *= dims[i];
    }
    stride = 1;
    std::vector<int64_t> strides(dims.size());
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) { strides[i] = stride; stride *= dims[i]; }
    for (size_t i = 0; i < dims.size(); ++i) {
        if (std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(i))) {
            const int r = spec.reduced_rank++;
            spec.reduced_dims[r] = dims[i]; spec.reduced_strides[r] = strides[i];
            if (dims[i] && spec.reduction_count > std::numeric_limits<int64_t>::max() / dims[i]) return -1;
            spec.reduction_count *= dims[i];
            if (param->keepdims) expected.push_back(1);
        } else {
            const int k = spec.kept_rank++;
            spec.kept_dims[k] = dims[i]; spec.kept_strides[k] = strides[i];
            expected.push_back(dims[i]);
        }
    }
    if (param->out->dims().data() != expected) return -1;
    if (param->out->numel() == 0) return 0;
    cd::DeviceBuffer<float> input, output;
    if (!cd::IsOutputReady<DataType::FP32>(param->out.get(), &expected) ||
        (spec.reduction_count && (!cd::IsTensorReady<DataType::FP32>(param->data.get()) ||
                                 cd::CopyTensorToDevice(param->data.get(), &input) != 0)) ||
        cd::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    constexpr int threads = 128;
    ReduceMaxCuda<<<static_cast<int>(std::min<int64_t>(65535, cd::DivUp(param->out->numel(), threads / 32))), threads, 0,
                    cd::InferenceStream()>>>(reinterpret_cast<const uint32_t*>(input.get()),
                                             reinterpret_cast<uint32_t*>(output.get()), param->out->numel(), spec);
    if (cd::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cd::CopyDeviceToTensor(&output, param->out.get());
}

struct GatherSpec { cd::CudaShape data; cd::CudaShape indices; int axis{0}; };

template <typename Bits>
__global__ void GatherElementsCuda(const Bits* input, const int64_t* indices, Bits* output,
                                    int64_t count, GatherSpec spec, int* error) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t selected = indices[i];
        if (selected < 0) selected += spec.data.dims[spec.axis];
        if (selected < 0 || selected >= spec.data.dims[spec.axis]) { atomicExch(error, 1); continue; }
        int64_t remaining = i, source = selected * spec.data.strides[spec.axis];
        for (int axis = 0; axis < spec.indices.rank; ++axis) {
            const int64_t coordinate = remaining / spec.indices.strides[axis];
            remaining %= spec.indices.strides[axis];
            if (axis != spec.axis) source += coordinate * spec.data.strides[axis];
        }
        output[i] = input[source];
    }
}

template <DataType dtype>
int RunGatherElements(YoloSelectionParam* param) {
    using T = cd::StorageT<dtype>;
    if (!param || !cd::IsTensorReady<dtype>(param->data.get()) ||
        !cd::IsTensorReady<DataType::INT64>(param->control.get()) ||
        !cd::IsOutputReady<dtype>(param->out.get(), &param->control->dims().data())) return -1;
    GatherSpec spec;
    if (!cd::MakeCudaShape(param->data->dims().data(), &spec.data) ||
        !cd::MakeCudaShape(param->control->dims().data(), &spec.indices) ||
        spec.data.rank != spec.indices.rank) return -1;
    const int64_t axis = param->axis < 0 ? param->axis + spec.data.rank : param->axis;
    if (axis < 0 || axis >= spec.data.rank) return -1;
    spec.axis = static_cast<int>(axis);
    for (int i = 0; i < spec.data.rank; ++i) {
        if (i != spec.axis && spec.indices.dims[i] > spec.data.dims[i]) return -1;
    }
    cd::DeviceBuffer<T> input, output;
    cd::DeviceBuffer<int64_t> indices;
    DeviceError error;
    if (cd::CopyTensorToDevice(param->data.get(), &input) != 0 ||
        cd::CopyTensorToDevice(param->control.get(), &indices) != 0 ||
        cd::AllocateTensorOnDevice(param->out.get(), &output) != 0 || error.Init() != 0) return -1;
    const int blocks = static_cast<int>(std::min<int64_t>(65535, cd::DivUp(param->out->numel(), 256)));
    // uint32_t aliases FP32 storage without passing values through float registers.
    if constexpr (dtype == DataType::FP32) {
        GatherElementsCuda<<<blocks, 256, 0, cd::InferenceStream()>>>(
            reinterpret_cast<const uint32_t*>(input.get()), indices.get(),
            reinterpret_cast<uint32_t*>(output.get()), param->out->numel(), spec, error.get());
    } else {
        GatherElementsCuda<<<blocks, 256, 0, cd::InferenceStream()>>>(input.get(), indices.get(), output.get(),
                                                                  param->out->numel(), spec, error.get());
    }
    if (cd::CudaCheck(cudaGetLastError()) != 0 || error.Check() != 0) return -1;
    return cd::CopyDeviceToTensor(&output, param->out.get());
}

struct BinarySpec { cd::CudaShape lhs; cd::CudaShape rhs; cd::CudaShape out; };

__device__ int64_t BroadcastIndex(int64_t linear, const cd::CudaShape& input, const cd::CudaShape& output) {
    int64_t offset = 0;
    for (int axis = output.rank - 1; axis >= 0; --axis) {
        const int64_t coordinate = linear % output.dims[axis];
        linear /= output.dims[axis];
        const int i = axis - (output.rank - input.rank);
        if (i >= 0 && input.dims[i] != 1) offset += coordinate * input.strides[i];
    }
    return offset;
}

__global__ void IntegerBinaryCuda(const int64_t* lhs, const int64_t* rhs, int64_t* output, int64_t count,
                                   BinarySpec spec, bool divide, bool fmod, int* error) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t a = lhs[BroadcastIndex(i, spec.lhs, spec.out)];
        const int64_t b = rhs[BroadcastIndex(i, spec.rhs, spec.out)];
        const bool overflow = a == (-9223372036854775807LL - 1) && b == -1;
        if (b == 0 || (divide && overflow)) { atomicExch(error, 1); continue; }
        int64_t value = divide ? a / b : (overflow ? 0 : a % b);
        if (!divide && !fmod && value != 0 && ((value < 0) != (b < 0))) value += b;
        output[i] = value;
    }
}

int RunIntegerBinary(Tensor* lhs, Tensor* rhs, Tensor* out, bool divide, bool fmod) {
    std::vector<int64_t> shape;
    if (!cd::IsTensorReady<DataType::INT64>(lhs) || !cd::IsTensorReady<DataType::INT64>(rhs) ||
        !cd::InferBroadcastShape(lhs->dims().data(), rhs->dims().data(), &shape) ||
        !cd::IsOutputReady<DataType::INT64>(out, &shape)) return -1;
    BinarySpec spec;
    if (!cd::MakeCudaShape(lhs->dims().data(), &spec.lhs) || !cd::MakeCudaShape(rhs->dims().data(), &spec.rhs) ||
        !cd::MakeCudaShape(shape, &spec.out)) return -1;
    cd::DeviceBuffer<int64_t> a, b, output;
    DeviceError error;
    if (cd::CopyTensorToDevice(lhs, &a) != 0 || cd::CopyTensorToDevice(rhs, &b) != 0 ||
        cd::AllocateTensorOnDevice(out, &output) != 0 || error.Init() != 0) return -1;
    IntegerBinaryCuda<<<static_cast<int>(std::min<int64_t>(65535, cd::DivUp(out->numel(), 256))), 256, 0,
                        cd::InferenceStream()>>>(a.get(), b.get(), output.get(), out->numel(), spec,
                                                 divide, fmod, error.get());
    if (cd::CudaCheck(cudaGetLastError()) != 0 || error.Check() != 0) return -1;
    return cd::CopyDeviceToTensor(&output, out);
}

class CudaYoloSelectionKernel : public KernelBase {
 public:
    Status Supports() const override {
        const auto* p = ParamAs<YoloSelectionParam>();
        if (!p || p->kind == operators::YoloSelectionKind::kTopK) {
            return Status::Error(StatusCode::kUnsupported, "CUDA selection requires compatible parameters");
        }
        for (const auto* tensor : {p->data.get(), p->out.get()}) {
            const auto status = SelectionShapeSupport(tensor);
            if (!status.ok()) return status;
        }
        if (p->control) return SelectionShapeSupport(p->control.get());
        return Status::Ok();
    }
    int32_t compute() override {
        auto* p = ParamAs<YoloSelectionParam>();
        if (!p || !p->data || !p->out) return -1;
        if (p->out->numel() == 0) return 0;
        switch (p->kind) {
            case operators::YoloSelectionKind::kReduceMax: return RunReduceMax(p);
            case operators::YoloSelectionKind::kGatherElements:
                if (p->data->data_type() == DataType::FP32) return RunGatherElements<DataType::FP32>(p);
                if (p->data->data_type() == DataType::INT64) return RunGatherElements<DataType::INT64>(p);
                return -1;
            case operators::YoloSelectionKind::kMod:
                return RunIntegerBinary(p->data.get(), p->control.get(), p->out.get(), false, p->fmod);
            default: return -1;
        }
    }
};

class CudaIntegerDivKernel : public KernelBase {
 public:
    Status Supports() const override {
        const auto* p = ParamAs<operators::BinaryParam>();
        if (!p) return Status::Error(StatusCode::kUnsupported, "CUDA integer Div requires binary parameters");
        for (const auto* tensor : {p->lhs.get(), p->rhs.get(), p->out.get()}) {
            const auto status = SelectionShapeSupport(tensor);
            if (!status.ok()) return status;
        }
        return Status::Ok();
    }
    int32_t compute() override {
        auto* p = ParamAs<operators::BinaryParam>();
        if (!p || !p->out) return -1;
        if (p->out->numel() == 0) return 0;
        return RunIntegerBinary(p->lhs.get(), p->rhs.get(), p->out.get(), true, false);
    }
};
}  // namespace

void EnsureCudaYoloSelectionKernelsRegistered() {
    auto& dispatcher = KernelDispatcher::instance();
    static const bool registered = [&dispatcher] {
        for (const auto dtype : {DataType::FP32, DataType::INT64}) {
            dispatcher.RegisterCandidate(DeviceType::CUDA, dtype, DataLayout::ND, "GatherElements",
                "cuda_exact_gather_elements", [] {
                return std::make_unique<CudaYoloSelectionKernel>();
            }, SelectionRequestSupport);
        }
        dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::FP32, DataLayout::ND, "ReduceMax",
            "cuda_fp32_warp_reduce_max", [] {
            return std::make_unique<CudaYoloSelectionKernel>();
        }, SelectionRequestSupport);
        dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::INT64, DataLayout::ND, "Mod",
            "cuda_exact_int64_mod", [] {
            return std::make_unique<CudaYoloSelectionKernel>();
        }, SelectionRequestSupport);
        dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::INT64, DataLayout::ND, "Div",
            "cuda_exact_int64_div", [] {
            return std::make_unique<CudaIntegerDivKernel>();
        }, SelectionRequestSupport);
        return true;
    }();
    (void)registered;
}
}  // namespace kernel
}  // namespace feather
