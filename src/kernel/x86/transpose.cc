#include "src/kernel/transpose.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

template <DataType dtype>
int32_t ComputeTransposeFallback(feather::operators::TransposeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);

    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        std::fill(in_coords.begin(), in_coords.end(), 0);
        for (size_t axis = 0; axis < param->perm.size(); ++axis) {
            in_coords[param->perm[axis]] = out_coords[axis];
        }
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            input_offset += in_coords[axis] * in_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

bool Is2DTranspose(const feather::operators::TransposeParam* param) {
    return param != nullptr && param->input != nullptr &&
           param->input->dims().size() == 2 && param->perm.size() == 2 &&
           param->perm[0] == 1 && param->perm[1] == 0;
}

bool IsYolov5HeadTranspose(const feather::operators::TransposeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return false;
    }
    const auto& in_dims = param->input->dims().data();
    if (in_dims.size() != 5 || param->perm.size() != 5) {
        return false;
    }
    return param->perm[0] == 0 && param->perm[1] == 1 &&
           param->perm[2] == 3 && param->perm[3] == 4 &&
           param->perm[4] == 2;
}

template <typename T>
void Transpose2DTiled(const T* input, T* output, int64_t rows, int64_t cols) {
    constexpr int64_t tile = 8;
    for (int64_t row_block = 0; row_block < rows; row_block += tile) {
        for (int64_t col_block = 0; col_block < cols; col_block += tile) {
            const int64_t row_end = std::min(row_block + tile, rows);
            const int64_t col_end = std::min(col_block + tile, cols);
            for (int64_t row = row_block; row < row_end; ++row) {
                for (int64_t col = col_block; col < col_end; ++col) {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
        }
    }
}

int32_t ComputeTransposeYolov5HeadFP16(feather::operators::TransposeParam* param) {
    if (!IsYolov5HeadTranspose(param) || param->input->data_type() != DataType::FP16) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const int64_t batch = in_dims[0];
    const int64_t anchors = in_dims[1];
    const int64_t channels = in_dims[2];
    const int64_t height = in_dims[3];
    const int64_t width = in_dims[4];
    const int64_t spatial = height * width;

    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    param->out->set_data_type(DataType::FP16);

    constexpr int64_t spatial_tile = 32;
    alignas(16) uint16_t lane_buffer[8];

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t a = 0; a < anchors; ++a) {
            const uint16_t* anchor_input = input + ((n * anchors + a) * channels * spatial);
            uint16_t* anchor_output = output + ((n * anchors + a) * spatial * channels);

            int64_t spatial_base = 0;
            for (; spatial_base + spatial_tile <= spatial; spatial_base += spatial_tile) {
                for (int64_t channel_base = 0; channel_base < channels; channel_base += 8) {
                    const int64_t channel_count = std::min<int64_t>(8, channels - channel_base);
                    if (channel_count == 8) {
                        for (int64_t s = 0; s < spatial_tile; ++s) {
                            for (int64_t c = 0; c < 8; ++c) {
                                lane_buffer[c] = anchor_input[(channel_base + c) * spatial + spatial_base + s];
                            }
                            const __m128i lane =
                                _mm_load_si128(reinterpret_cast<const __m128i*>(lane_buffer));
                            _mm_storeu_si128(
                                reinterpret_cast<__m128i*>(anchor_output + (spatial_base + s) * channels + channel_base),
                                lane);
                        }
                    } else {
                        for (int64_t s = 0; s < spatial_tile; ++s) {
                            for (int64_t c = 0; c < channel_count; ++c) {
                                anchor_output[(spatial_base + s) * channels + channel_base + c] =
                                    anchor_input[(channel_base + c) * spatial + spatial_base + s];
                            }
                        }
                    }
                }
            }

            for (; spatial_base < spatial; ++spatial_base) {
                for (int64_t c = 0; c < channels; ++c) {
                    anchor_output[spatial_base * channels + c] = anchor_input[c * spatial + spatial_base];
                }
            }
        }
    }
    return 0;
}

template <typename T>
int32_t ComputeTransposeRaw(feather::operators::TransposeParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const T* input = param->input->data<T>();
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);

    if (Is2DTranspose(param)) {
        const int64_t rows = in_dims[0];
        const int64_t cols = in_dims[1];
        Transpose2DTiled(input, output, rows, cols);
        return 0;
    }

    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        std::fill(in_coords.begin(), in_coords.end(), 0);
        for (size_t axis = 0; axis < param->perm.size(); ++axis) {
            in_coords[param->perm[axis]] = out_coords[axis];
        }
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            input_offset += in_coords[axis] * in_strides[axis];
        }
        output[linear] = input[input_offset];
    }
    return 0;
}

}  // namespace

template <>
int32_t TransposeKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Transpose::FP32");
    auto* param = static_cast<feather::operators::TransposeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeTransposeFallback<DataType::FP32>(param);
    }
    return ComputeTransposeRaw<float>(param, DataType::FP32);
}

template <>
int32_t TransposeKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Transpose::FP16");
    auto* param = static_cast<feather::operators::TransposeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeTransposeFallback<DataType::FP16>(param);
    }
    if (ComputeTransposeYolov5HeadFP16(param) == 0) {
        return 0;
    }
    return ComputeTransposeRaw<uint16_t>(param, DataType::FP16);
}

void EnsureX86TransposeKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
