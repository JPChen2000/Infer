#include "src/kernel/yolo_decode.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "util/fp16.h"
#include "util/thread_pool_nv.h"
#include "util/threading.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

inline float Sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

size_t GetYoloDecodeThreadCount(int64_t total_work_items) {
    return ThreadCountForWorkItems(total_work_items);
}

ThreadPoolNv& GetYoloDecodeThreadPool() {
    static ThreadPoolNv pool(DefaultThreadCount());
    return pool;
}

template <typename Fn>
void ParallelForYoloDecode(int64_t total_work_items, Fn&& fn) {
    if (total_work_items <= 1 || total_work_items < 4096) {
        fn(0, total_work_items);
        return;
    }

    const size_t thread_count = GetYoloDecodeThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetYoloDecodeThreadPool();
    std::vector<std::future<int>> futures;
    futures.reserve(thread_count);

    const int64_t chunk_size =
        (total_work_items + static_cast<int64_t>(thread_count) - 1) / static_cast<int64_t>(thread_count);
    for (size_t tid = 0; tid < thread_count; ++tid) {
        const int64_t begin = static_cast<int64_t>(tid) * chunk_size;
        const int64_t end = std::min(total_work_items, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.emplace_back(pool.enqueue([begin, end, &fn](int) {
            fn(begin, end);
            return 0;
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

template <typename T>
inline float LoadValue(const T* data, int64_t index) {
    return static_cast<float>(data[index]);
}

template <>
inline float LoadValue<uint16_t>(const uint16_t* data, int64_t index) {
    return HalfToFloat(data[index]);
}

template <typename T>
inline void StoreValue(T* data, int64_t index, float value) {
    data[index] = static_cast<T>(value);
}

template <>
inline void StoreValue<uint16_t>(uint16_t* data, int64_t index, float value) {
    data[index] = FloatToHalf(value);
}

template <DataType dtype>
int32_t ComputeYoloDecodeFallback(feather::operators::YoloDecodeParam* param) {
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr ||
        param->grid == nullptr || param->stride == nullptr || param->wh_scale == nullptr ||
        param->anchor_grid == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& grid_dims = param->grid->dims().data();
    if (in_dims.size() != 4 || grid_dims.size() != 5) {
        return -1;
    }

    ImageShape4D input_shape;
    if (!DecodeImageShape4D(in_dims, param->input->layout(), &input_shape)) {
        return -1;
    }
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    const int64_t batch = input_shape.n;
    const int64_t channels = input_shape.c;
    const int64_t height = input_shape.h;
    const int64_t width = input_shape.w;
    const int64_t anchors = grid_dims[1];
    const int64_t attrs = channels / anchors;
    if (anchors <= 0 || attrs < 5 || channels % anchors != 0) {
        return -1;
    }

    const float xy_scale = TensorIO<dtype>::Read(param->xy_scale.get(), 0);
    const float stride = TensorIO<dtype>::Read(param->stride.get(), 0);
    const float wh_scale = TensorIO<dtype>::Read(param->wh_scale.get(), 0);
    const int64_t grid_batch = grid_dims[0];
    param->out->set_data_type(dtype);

    for (int64_t n = 0; n < batch; ++n) {
        const int64_t grid_n = grid_batch == 1 ? 0 : n;
        for (int64_t anchor = 0; anchor < anchors; ++anchor) {
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t point = ((anchor * height + y) * width + x);
                    const int64_t out_base = (n * anchors * height * width + point) * attrs;
                    for (int64_t attr = 0; attr < attrs; ++attr) {
                        const int64_t channel = anchor * attrs + attr;
                        const int64_t input_offset =
                            OffsetForImage4D(layout, n, channel, y, x, channels, height, width);
                        const float value = Sigmoid(TensorIO<dtype>::Read(param->input.get(), input_offset));
                        float decoded = value;
                        if (attr < 2) {
                            const int64_t grid_offset =
                                ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + attr;
                            decoded = (value * xy_scale + TensorIO<dtype>::Read(param->grid.get(), grid_offset)) *
                                      stride;
                        } else if (attr < 4) {
                            const int64_t anchor_offset =
                                ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + (attr - 2);
                            const float scaled = value * wh_scale;
                            decoded = scaled * scaled * TensorIO<dtype>::Read(param->anchor_grid.get(), anchor_offset);
                        }
                        TensorIO<dtype>::Write(param->out.get(), out_base + attr, decoded);
                    }
                }
            }
        }
    }
    return 0;
}

template <typename T>
int32_t ComputeYoloDecodeRaw(feather::operators::YoloDecodeParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr ||
        param->grid == nullptr || param->stride == nullptr || param->wh_scale == nullptr ||
        param->anchor_grid == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& grid_dims = param->grid->dims().data();
    if (in_dims.size() != 4 || grid_dims.size() != 5) {
        return -1;
    }

    ImageShape4D input_shape;
    if (!DecodeImageShape4D(in_dims, param->input->layout(), &input_shape)) {
        return -1;
    }
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    const int64_t batch = input_shape.n;
    const int64_t channels = input_shape.c;
    const int64_t height = input_shape.h;
    const int64_t width = input_shape.w;
    const int64_t anchors = grid_dims[1];
    const int64_t attrs = channels / anchors;
    if (anchors <= 0 || attrs < 5 || channels % anchors != 0) {
        return -1;
    }

    const T* input = param->input->data<T>();
    const T* xy_scale_ptr = param->xy_scale->data<T>();
    const T* grid = param->grid->data<T>();
    const T* stride_ptr = param->stride->data<T>();
    const T* wh_scale_ptr = param->wh_scale->data<T>();
    const T* anchor_grid = param->anchor_grid->data<T>();
    param->out->set_data_type(dtype);
    T* output = param->out->mutable_data<T>();
    if (input == nullptr || xy_scale_ptr == nullptr || grid == nullptr || stride_ptr == nullptr ||
        wh_scale_ptr == nullptr || anchor_grid == nullptr || output == nullptr) {
        return -1;
    }

    const float xy_scale = LoadValue(xy_scale_ptr, 0);
    const float stride = LoadValue(stride_ptr, 0);
    const float wh_scale = LoadValue(wh_scale_ptr, 0);
    const int64_t grid_batch = grid_dims[0];
    const int64_t points_per_batch = anchors * height * width;
    const int64_t total_points = batch * points_per_batch;
    const int64_t spatial = height * width;

    ParallelForYoloDecode(total_points, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t n = index / points_per_batch;
            const int64_t point_index = index % points_per_batch;
            const int64_t anchor = point_index / spatial;
            const int64_t spatial_index = point_index % spatial;
            const int64_t y = spatial_index / width;
            const int64_t x = spatial_index % width;
            const int64_t grid_n = grid_batch == 1 ? 0 : n;

            const int64_t out_base = index * attrs;
            const int64_t grid_base = (((grid_n * anchors + anchor) * height + y) * width + x) * 2;

            for (int64_t attr = 0; attr < attrs; ++attr) {
                const int64_t channel = anchor * attrs + attr;
                const float value =
                    Sigmoid(LoadValue(input, OffsetForImage4D(layout, n, channel, y, x, channels, height, width)));
                float decoded = value;
                if (attr < 2) {
                    decoded = (value * xy_scale + LoadValue(grid, grid_base + attr)) * stride;
                } else if (attr < 4) {
                    const float scaled = value * wh_scale;
                    decoded = scaled * scaled * LoadValue(anchor_grid, grid_base + (attr - 2));
                }
                StoreValue(output, out_base + attr, decoded);
            }
        }
    });

    return 0;
}

}  // namespace

template <>
int32_t YoloDecodeKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::YoloDecode::FP32");
    auto* param = static_cast<feather::operators::YoloDecodeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr || param->grid == nullptr ||
        param->stride == nullptr || param->wh_scale == nullptr || param->anchor_grid == nullptr ||
        param->out == nullptr || param->input->data_type() != DataType::FP32 ||
        param->xy_scale->data_type() != DataType::FP32 || param->grid->data_type() != DataType::FP32 ||
        param->stride->data_type() != DataType::FP32 || param->wh_scale->data_type() != DataType::FP32 ||
        param->anchor_grid->data_type() != DataType::FP32) {
        return ComputeYoloDecodeFallback<DataType::FP32>(param);
    }
    return ComputeYoloDecodeRaw<float>(param, DataType::FP32);
}

template <>
int32_t YoloDecodeKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::YoloDecode::FP16");
    auto* param = static_cast<feather::operators::YoloDecodeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr || param->grid == nullptr ||
        param->stride == nullptr || param->wh_scale == nullptr || param->anchor_grid == nullptr ||
        param->out == nullptr || param->input->data_type() != DataType::FP16 ||
        param->xy_scale->data_type() != DataType::FP16 || param->grid->data_type() != DataType::FP16 ||
        param->stride->data_type() != DataType::FP16 || param->wh_scale->data_type() != DataType::FP16 ||
        param->anchor_grid->data_type() != DataType::FP16) {
        return ComputeYoloDecodeFallback<DataType::FP16>(param);
    }
    return ComputeYoloDecodeRaw<uint16_t>(param, DataType::FP16);
}

void EnsureX86YoloDecodeKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "YoloDecode",
            []() { return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "YoloDecode",
            []() { return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
