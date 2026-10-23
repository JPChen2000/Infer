// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

// CUB must precede Tensor's legacy BYTE_LEN macro (CUDA 11.8).
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/kernel/cuda/nms.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/non_max_suppression_op.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {
namespace cd = cuda_detail;

class NmsWorkspace {
 public:
    ~NmsWorkspace() { cd::ReleaseTemporaryDeviceBuffer(data_, bytes_); }
    NmsWorkspace() = default;
    NmsWorkspace(const NmsWorkspace&) = delete;
    NmsWorkspace& operator=(const NmsWorkspace&) = delete;
    bool Allocate(size_t bytes) {
        if (cd::AcquireTemporaryDeviceBuffer(bytes, &data_) != 0) {
            std::fprintf(stderr, "CUDA NMS: unable to allocate %zu bytes of session workspace\n", bytes);
            return false;
        }
        bytes_ = bytes;
        return true;
    }
    template <typename T> T* As() const { return static_cast<T*>(data_); }
 private:
    void* data_{nullptr};
    size_t bytes_{0};
};

bool CheckedProduct(int64_t a, int64_t b, int64_t* product) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b)) return false;
    *product = a * b;
    return true;
}

bool ValidNmsDimensions(const cd::CudaNmsOptions& options) {
    if (options.batches < 0 || options.classes < 0 || options.boxes < 0 || options.boxes > INT_MAX ||
        options.max_per_class < 0) return false;
    int64_t box_count = 0, score_count = 0, segments = 0;
    size_t bytes = 0;
    return CheckedProduct(options.batches, options.boxes, &box_count) &&
           CheckedProduct(box_count, 4, &box_count) && cd::SafeByteCount(box_count, sizeof(float), &bytes) &&
           CheckedProduct(options.batches, options.classes, &segments) &&
           CheckedProduct(segments, options.boxes, &score_count) &&
           cd::SafeByteCount(score_count, sizeof(float), &bytes) &&
           CheckedProduct(score_count, 3, &score_count) &&
           cd::SafeByteCount(score_count, sizeof(int64_t), &bytes);
}

__global__ void PrepareNmsKeys(const float* boxes, const float* scores, int count,
                               float threshold, bool inclusive, uint32_t* keys, int* order) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float score = scores[index];
        bool valid = !isnan(score) && (inclusive ? score >= threshold : score > threshold);
        for (int coordinate = 0; coordinate < 4; ++coordinate) {
            valid = valid && isfinite(boxes[index * 4 + coordinate]);
        }
        const uint32_t bits = __float_as_uint(score);
        // All finite keys exceed zero, reserved for invalid candidates. Both
        // zero signs share a key so CUB's stable sort preserves source ties.
        keys[index] = !valid ? 0u : (bits & 0x7fffffffu) == 0 ? 0x80000000u :
                      (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        order[index] = static_cast<int>(index);
    }
}

__global__ void CountSortedNmsCandidates(const uint32_t* keys, int count, int* valid_count) {
    // Valid keys form a sorted prefix. Only this scalar crosses to the host
    // before allocating masks, so low score boxes do not consume mask space.
    int begin = 0, end = count;
    while (begin < end) {
        const int middle = begin + (end - begin) / 2;
        if (keys[middle] != 0) begin = middle + 1;
        else end = middle;
    }
    *valid_count = begin;
}

template <typename T> struct NmsBox { T first0, first1, last0, last1; };

template <typename T>
__device__ NmsBox<T> LoadNmsBox(const float* input, int source, bool center, bool preserve_order) {
    const int64_t offset = static_cast<int64_t>(source) * 4;
    T a = static_cast<T>(input[offset]), b = static_cast<T>(input[offset + 1]);
    T c = static_cast<T>(input[offset + 2]), d = static_cast<T>(input[offset + 3]);
    if (center) {
        const T half_width = c * T(0.5), half_height = d * T(0.5);
        c = a + half_width; d = b + half_height;
        a -= half_width; b -= half_height;
        return {a, b, c, d};
    }
    if (preserve_order) return {a, b, c, d};
    return {a < c ? a : c, b < d ? b : d, a > c ? a : c, b > d ? b : d};
}

template <typename T>
__device__ bool NmsOverlap(const NmsBox<T>& a, const NmsBox<T>& b, T threshold) {
    const T first0 = a.first0 > b.first0 ? a.first0 : b.first0;
    const T first1 = a.first1 > b.first1 ? a.first1 : b.first1;
    const T last0 = a.last0 < b.last0 ? a.last0 : b.last0;
    const T last1 = a.last1 < b.last1 ? a.last1 : b.last1;
    const T width = last0 > first0 ? last0 - first0 : T(0);
    const T height = last1 > first1 ? last1 - first1 : T(0);
    const T intersection = width * height;
    const T area_a = (a.last0 > a.first0 ? a.last0 - a.first0 : T(0)) *
                     (a.last1 > a.first1 ? a.last1 - a.first1 : T(0));
    const T area_b = (b.last0 > b.first0 ? b.last0 - b.first0 : T(0)) *
                     (b.last1 > b.first1 ? b.last1 - b.first1 : T(0));
    // Separate rounded operations avoid contraction changing a threshold tie.
    T union_area;
    if constexpr (std::is_same<T, double>::value) union_area = __dsub_rn(__dadd_rn(area_a, area_b), intersection);
    else union_area = __fsub_rn(__fadd_rn(area_a, area_b), intersection);
    return union_area > T(0) && intersection / union_area > threshold;
}

template <typename T>
__global__ void BuildNmsMasks(const float* boxes, const int64_t* labels, const int* order,
                              int count, int words, bool center, bool preserve_order, float threshold,
                              unsigned long long* masks) {
    constexpr int kTile = 64;
    __shared__ NmsBox<T> columns[kTile];
    __shared__ int64_t column_labels[kTile];
    const int lane = threadIdx.x;
    const int64_t tiles = static_cast<int64_t>(words) * words;
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int row_tile = static_cast<int>(tile / words);
        const int column_tile = static_cast<int>(tile % words);
        if (column_tile < row_tile) continue;
        const int64_t column_index = static_cast<int64_t>(column_tile) * kTile + lane;
        if (column_index < count) {
            const int source = order[column_index];
            columns[lane] = LoadNmsBox<T>(boxes, source, center, preserve_order);
            if (labels) column_labels[lane] = labels[source];
        }
        __syncthreads();
        const int64_t row = static_cast<int64_t>(row_tile) * kTile + lane;
        if (row < count) {
            const int source = order[row];
            const auto box = LoadNmsBox<T>(boxes, source, center, preserve_order);
            const int64_t label = labels ? labels[source] : 0;
            const int remaining = count - column_tile * kTile;
            const int end = remaining < kTile ? remaining : kTile;
            const int begin = row_tile == column_tile ? lane + 1 : 0;
            unsigned long long mask = 0;
            for (int column = begin; column < end; ++column) {
                if ((!labels || label == column_labels[column]) &&
                    NmsOverlap(box, columns[column], static_cast<T>(threshold))) {
                    mask |= 1ull << column;
                }
            }
            masks[row * words + column_tile] = mask;
        }
        __syncthreads();
    }
}

__global__ void GreedyNms(const unsigned long long* masks, const int* order,
                          int count, int words, int maximum, int64_t batch, int64_t cls,
                          int64_t* selected, int* selected_count) {
    extern __shared__ unsigned long long removed[];
    __shared__ int keep;
    __shared__ int retained;
    if (threadIdx.x == 0) retained = 0;
    for (int word = threadIdx.x; word < words; word += blockDim.x) removed[word] = 0;
    __syncthreads();
    for (int64_t row = 0; row < count; ++row) {
        const int row_word = static_cast<int>(row / 64);
        if (threadIdx.x == 0) {
            keep = (removed[row_word] & (1ull << (row % 64))) == 0;
            if (keep) {
                const int64_t output = static_cast<int64_t>(retained) * 3;
                selected[output] = batch;
                selected[output + 1] = cls;
                selected[output + 2] = order[row];
                ++retained;
            }
        }
        __syncthreads();
        if (retained >= maximum) break;
        if (keep) {
            for (int word = row_word + threadIdx.x; word < words; word += blockDim.x) {
                removed[word] |= masks[row * words + word];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *selected_count = retained;
}

Status NmsShapeSupport(const Tensor* boxes, const Tensor* scores) {
    if (!boxes || !scores || boxes->data_type() != DataType::FP32 || scores->data_type() != DataType::FP32 ||
        boxes->dims().size() != 3 || scores->dims().size() != 3) {
        return Status::Error(StatusCode::kUnsupported, "CUDA NMS requires rank-three FP32 boxes and scores");
    }
    const auto& b = boxes->dims().data();
    const auto& s = scores->dims().data();
    cd::CudaNmsOptions options;
    options.batches = b[0]; options.boxes = b[1]; options.classes = s[1];
    if (b[2] != 4 || s[0] != b[0] || s[2] != b[1] || !ValidNmsDimensions(options)) {
        return Status::Error(StatusCode::kUnsupported, "CUDA NMS shape exceeds safe indexing or CUB's INT_MAX box limit");
    }
    return Status::Ok();
}

class CudaNonMaxSuppressionKernel final : public KernelBase {
 public:
    Status Supports() const override {
        const auto* p = ParamAs<operators::NmsParam>();
        if (!p || !operators::ValidateNms(*p)) {
            return Status::Error(StatusCode::kUnsupported, "invalid CUDA NonMaxSuppression parameters");
        }
        return NmsShapeSupport(p->boxes.get(), p->scores.get());
    }
    int32_t compute() override {
        AutoTimer timer("CUDA::NonMaxSuppression::FP32");
        auto* p = ParamAs<operators::NmsParam>();
        if (!p || !Supports().ok()) return -1;
        for (const auto& control : {p->max_output, p->iou_threshold, p->score_threshold}) {
            if (control && !control->is_immutable()) {
                size_t bytes = 0;
                if (!cd::SafeByteCount(control->numel(), DataTypeBytes(control->data_type()), &bytes) ||
                    cd::SyncTensorToHostIfNeeded(control.get(), bytes, control->raw_data()) != 0) return -1;
            }
        }
        cd::CudaNmsOptions options;
        if (!operators::ReadNmsControls(*p, &options.max_per_class, &options.iou_threshold,
                                       &options.score_threshold)) return -1;
        options.batches = p->boxes->dims()[0];
        options.boxes = p->boxes->dims()[1];
        options.classes = p->scores->dims()[1];
        options.center_point_box = p->center_point_box != 0;
        // Match the Common kernel's arithmetic for finite FP32 coordinates,
        // including boxes whose areas exceed the finite FP32 range.
        options.double_iou = true;
        std::vector<int64_t> selected;
        if (options.batches && options.classes && options.boxes && options.max_per_class) {
            cd::DeviceBuffer<float> boxes, scores;
            if (cd::CopyTensorToDevice(p->boxes.get(), &boxes) != 0 ||
                cd::CopyTensorToDevice(p->scores.get(), &scores) != 0 ||
                cd::SelectNmsOnCuda(boxes.get(), scores.get(), nullptr, options, &selected) != 0) return -1;
        }
        if (!operators::FinishNmsOutput(*p, selected)) return -1;
        // This terminal dynamic output is already on the host. A cached device
        // allocation from an earlier run must never overwrite its new contents.
        cd::InvalidateTensorDevice(p->out.get());
        return 0;
    }
};
}  // namespace

namespace cuda_detail {
int SelectNmsOnCuda(const float* boxes, const float* scores, const int64_t* labels,
                    const CudaNmsOptions& options, std::vector<int64_t>* selected) {
    if (!selected) return -1;
    selected->clear();
    if (!ValidNmsDimensions(options) || !std::isfinite(options.iou_threshold) ||
        options.iou_threshold < 0 || options.iou_threshold > 1 || !std::isfinite(options.score_threshold)) return -1;
    if (!options.batches || !options.classes || !options.boxes || !options.max_per_class) return 0;
    if (!boxes || !scores) return -1;
    const int count = static_cast<int>(options.boxes);
    const auto stream = InferenceStream();
    NmsWorkspace keys_in, keys_out, order_in, order_out, counter, temporary;
    size_t key_bytes = 0, order_bytes = 0;
    if (!SafeByteCount(count, sizeof(uint32_t), &key_bytes) || !SafeByteCount(count, sizeof(int), &order_bytes) ||
        !keys_in.Allocate(key_bytes) || !keys_out.Allocate(key_bytes) ||
        !order_in.Allocate(order_bytes) || !order_out.Allocate(order_bytes) || !counter.Allocate(sizeof(int))) return -1;
    size_t temporary_bytes = 0;
    if (CudaCheck(cub::DeviceRadixSort::SortPairsDescending(nullptr, temporary_bytes,
            keys_in.As<uint32_t>(), keys_out.As<uint32_t>(), order_in.As<int>(), order_out.As<int>(),
            count, 0, 32, stream)) != 0 || !temporary.Allocate(temporary_bytes)) return -1;
    const int blocks = static_cast<int>(std::min<int64_t>(DivUp(count, kCudaThreads), 65535));
    try {
        for (int64_t batch = 0; batch < options.batches; ++batch) {
            const float* batch_boxes = boxes + batch * options.boxes * 4;
            const int64_t* batch_labels = labels ? labels + batch * options.boxes : nullptr;
            for (int64_t cls = 0; cls < options.classes; ++cls) {
                const float* class_scores = scores + (batch * options.classes + cls) * options.boxes;
                PrepareNmsKeys<<<blocks, kCudaThreads, 0, stream>>>(batch_boxes, class_scores, count,
                    options.score_threshold, options.score_inclusive, keys_in.As<uint32_t>(), order_in.As<int>());
                if (CudaCheck(cudaGetLastError()) != 0 ||
                    CudaCheck(cub::DeviceRadixSort::SortPairsDescending(temporary.As<void>(), temporary_bytes,
                        keys_in.As<uint32_t>(), keys_out.As<uint32_t>(), order_in.As<int>(), order_out.As<int>(),
                        count, 0, 32, stream)) != 0) return -1;
                CountSortedNmsCandidates<<<1, 1, 0, stream>>>(keys_out.As<uint32_t>(), count, counter.As<int>());
                int valid = 0;
                if (CudaCheck(cudaGetLastError()) != 0 ||
                    CudaCheck(cudaMemcpyAsync(&valid, counter.As<int>(), sizeof(valid), cudaMemcpyDeviceToHost, stream)) != 0 ||
                    SynchronizeInferenceStream() != 0) return -1;
                if (valid == 0) continue;
                const int maximum = static_cast<int>(std::min<int64_t>(options.max_per_class, valid));
                const int words = static_cast<int>(DivUp(valid, 64));
                size_t mask_bytes = 0, output_bytes = 0;
                int64_t mask_count = 0;
                if (!CheckedProduct(valid, words, &mask_count) ||
                    !SafeByteCount(mask_count, sizeof(unsigned long long), &mask_bytes) ||
                    !SafeByteCount(static_cast<int64_t>(maximum) * 3, sizeof(int64_t), &output_bytes)) return -1;
                const size_t shared_bytes = static_cast<size_t>(words) * sizeof(unsigned long long);
                int device = 0, shared_limit = 0;
                if (CudaCheck(cudaGetDevice(&device)) != 0 || CudaCheck(cudaDeviceGetAttribute(
                        &shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device)) != 0) return -1;
                // Leave room for the greedy kernel's two static shared ints.
                if (shared_bytes + 2 * sizeof(int) > static_cast<size_t>(shared_limit)) {
                    std::fprintf(stderr, "CUDA NMS: %d candidates require %zu bytes of shared suppression workspace (limit %d)\n",
                                 valid, shared_bytes + 2 * sizeof(int), shared_limit);
                    return -1;
                }
                NmsWorkspace masks, output;
                if (!masks.Allocate(mask_bytes) || !output.Allocate(output_bytes)) return -1;
                const int tile_blocks = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(words) * words, 65535));
                if (options.double_iou) {
                    BuildNmsMasks<double><<<tile_blocks, 64, 0, stream>>>(batch_boxes, batch_labels,
                        order_out.As<int>(), valid, words, options.center_point_box, options.preserve_corner_order, options.iou_threshold,
                        masks.As<unsigned long long>());
                } else {
                    BuildNmsMasks<float><<<tile_blocks, 64, 0, stream>>>(batch_boxes, batch_labels,
                        order_out.As<int>(), valid, words, options.center_point_box, options.preserve_corner_order, options.iou_threshold,
                        masks.As<unsigned long long>());
                }
                if (CudaCheck(cudaGetLastError()) != 0) return -1;
                GreedyNms<<<1, kCudaThreads, shared_bytes, stream>>>(masks.As<unsigned long long>(),
                    order_out.As<int>(), valid, words, maximum, batch, cls, output.As<int64_t>(), counter.As<int>());
                int retained = 0;
                if (CudaCheck(cudaGetLastError()) != 0 ||
                    CudaCheck(cudaMemcpyAsync(&retained, counter.As<int>(), sizeof(retained), cudaMemcpyDeviceToHost, stream)) != 0 ||
                    SynchronizeInferenceStream() != 0 || retained < 0 || retained > maximum) return -1;
                const size_t old_size = selected->size();
                const size_t added = static_cast<size_t>(retained) * 3;
                if (added > selected->max_size() - old_size) return -1;
                selected->resize(old_size + added);
                if (added && (CudaCheck(cudaMemcpyAsync(selected->data() + old_size, output.As<int64_t>(),
                        added * sizeof(int64_t), cudaMemcpyDeviceToHost, stream)) != 0 ||
                        SynchronizeInferenceStream() != 0)) return -1;
            }
        }
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "CUDA NMS: unable to allocate retained host indices\n");
        return -1;
    } catch (const std::length_error&) {
        return -1;
    }
    return 0;
}
}  // namespace cuda_detail

void EnsureCudaNonMaxSuppressionKernelsRegistered() {
    auto& dispatcher = KernelDispatcher::instance();
    static const bool registered = [&dispatcher] {
        dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::FP32, DataLayout::ND,
            "NonMaxSuppression", "cuda_fp32_stable_radix_nms",
            [] { return std::make_unique<CudaNonMaxSuppressionKernel>(); },
            [](const KernelRequest& request) {
                if (request.tensors.empty()) return Status::Ok();
                if (request.tensors.size() < 2) return Status::Error(StatusCode::kUnsupported, "CUDA NMS requires boxes and scores");
                return NmsShapeSupport(request.tensors[0].get(), request.tensors[1].get());
            });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
