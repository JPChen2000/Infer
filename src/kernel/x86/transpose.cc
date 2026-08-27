#include "src/kernel/transpose.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/bf16.h"
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

bool IsPhysicalNoOpPermutation(const feather::operators::TransposeParam* param) {
    if (param == nullptr || param->input == nullptr) {
        return false;
    }

    const auto& input_dims = param->input->dims().data();
    if (param->perm.size() != input_dims.size()) {
        return false;
    }

    std::vector<bool> seen(input_dims.size(), false);
    size_t next_non_singleton_axis = 0;
    for (const int64_t source_axis : param->perm) {
        if (source_axis < 0 || source_axis >= static_cast<int64_t>(input_dims.size())) {
            return false;
        }
        const size_t axis = static_cast<size_t>(source_axis);
        if (seen[axis]) {
            return false;
        }
        seen[axis] = true;
        if (input_dims[axis] == 1) {
            continue;
        }

        while (next_non_singleton_axis < input_dims.size() && input_dims[next_non_singleton_axis] == 1) {
            ++next_non_singleton_axis;
        }
        if (next_non_singleton_axis == input_dims.size() || axis != next_non_singleton_axis) {
            return false;
        }
        ++next_non_singleton_axis;
    }
    return true;
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

bool IsLastTwoAxesTranspose(const feather::operators::TransposeParam* param) {
    if (param == nullptr || param->input == nullptr) {
        return false;
    }
    const size_t rank = param->input->dims().size();
    if (rank < 2 || param->perm.size() != rank) {
        return false;
    }
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
        if (param->perm[axis] != static_cast<int64_t>(axis)) {
            return false;
        }
    }
    return param->perm[rank - 2] == static_cast<int64_t>(rank - 1) &&
           param->perm[rank - 1] == static_cast<int64_t>(rank - 2);
}

bool IsVitMiddleAxesTranspose(const feather::operators::TransposeParam* param) {
    return param != nullptr && param->input != nullptr && param->out != nullptr &&
           param->input->dims().size() == 4 && param->perm == std::vector<int64_t>({0, 2, 1, 3});
}

bool IsVitHeadLastTranspose(const feather::operators::TransposeParam* param) {
    return param != nullptr && param->input != nullptr && param->out != nullptr &&
           param->input->dims().size() == 4 && param->perm == std::vector<int64_t>({0, 2, 3, 1});
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

    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        int64_t row_block = 0;
        for (; row_block + tile <= rows; row_block += tile) {
            int64_t col_block = 0;
            for (; col_block + tile <= cols; col_block += tile) {
                const T* row0 = input + row_block * cols + col_block;
                const T* row1 = row0 + cols;
                const T* row2 = row1 + cols;
                const T* row3 = row2 + cols;
                const T* row4 = row3 + cols;
                const T* row5 = row4 + cols;
                const T* row6 = row5 + cols;
                const T* row7 = row6 + cols;

                const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row0));
                const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1));
                const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row2));
                const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row3));
                const __m128i r4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row4));
                const __m128i r5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row5));
                const __m128i r6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row6));
                const __m128i r7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row7));

                const __m128i a0 = _mm_unpacklo_epi16(r0, r1);
                const __m128i a1 = _mm_unpackhi_epi16(r0, r1);
                const __m128i a2 = _mm_unpacklo_epi16(r2, r3);
                const __m128i a3 = _mm_unpackhi_epi16(r2, r3);
                const __m128i a4 = _mm_unpacklo_epi16(r4, r5);
                const __m128i a5 = _mm_unpackhi_epi16(r4, r5);
                const __m128i a6 = _mm_unpacklo_epi16(r6, r7);
                const __m128i a7 = _mm_unpackhi_epi16(r6, r7);

                const __m128i b0 = _mm_unpacklo_epi32(a0, a2);
                const __m128i b1 = _mm_unpackhi_epi32(a0, a2);
                const __m128i b2 = _mm_unpacklo_epi32(a1, a3);
                const __m128i b3 = _mm_unpackhi_epi32(a1, a3);
                const __m128i b4 = _mm_unpacklo_epi32(a4, a6);
                const __m128i b5 = _mm_unpackhi_epi32(a4, a6);
                const __m128i b6 = _mm_unpacklo_epi32(a5, a7);
                const __m128i b7 = _mm_unpackhi_epi32(a5, a7);

                T* output0 = output + col_block * rows + row_block;
                T* output1 = output0 + rows;
                T* output2 = output1 + rows;
                T* output3 = output2 + rows;
                T* output4 = output3 + rows;
                T* output5 = output4 + rows;
                T* output6 = output5 + rows;
                T* output7 = output6 + rows;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output0), _mm_unpacklo_epi64(b0, b4));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output1), _mm_unpackhi_epi64(b0, b4));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output2), _mm_unpacklo_epi64(b1, b5));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output3), _mm_unpackhi_epi64(b1, b5));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output4), _mm_unpacklo_epi64(b2, b6));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output5), _mm_unpackhi_epi64(b2, b6));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output6), _mm_unpacklo_epi64(b3, b7));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(output7), _mm_unpackhi_epi64(b3, b7));
            }
            for (; col_block < cols; ++col_block) {
                for (int64_t row = row_block; row < row_block + tile; ++row) {
                    output[col_block * rows + row] = input[row * cols + col_block];
                }
            }
        }
        for (; row_block < rows; ++row_block) {
            for (int64_t col = 0; col < cols; ++col) {
                output[col * rows + row_block] = input[row_block * cols + col];
            }
        }
        return;
    }

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

template <typename T>
void TransposeVitMiddleAxes(const T* input, T* output, int64_t batch, int64_t first, int64_t second,
                            int64_t inner) {
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t second_index = 0; second_index < second; ++second_index) {
            for (int64_t first_index = 0; first_index < first; ++first_index) {
                const T* input_row = input + ((n * first + first_index) * second + second_index) * inner;
                T* output_row = output + ((n * second + second_index) * first + first_index) * inner;
                std::memcpy(output_row, input_row, static_cast<size_t>(inner) * sizeof(T));
            }
        }
    }
}

template <typename T>
void TransposeVitHeadLast(const T* input, T* output, int64_t batch, int64_t first, int64_t second,
                          int64_t inner) {
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t second_index = 0; second_index < second; ++second_index) {
            for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
                T* output_row = output + ((n * second + second_index) * inner + inner_index) * first;
                for (int64_t first_index = 0; first_index < first; ++first_index) {
                    output_row[first_index] = input[((n * first + first_index) * second + second_index) * inner +
                                                    inner_index];
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
    param->out->set_data_type(dtype);
    if (IsPhysicalNoOpPermutation(param) && param->input->IsInitialized() && param->out->IsInitialized() &&
        param->input->raw_data() == param->out->raw_data()) {
        return 0;
    }
    const T* input = param->input->data<T>();
    T* output = static_cast<T*>(param->out->raw_data());

    if (IsVitMiddleAxesTranspose(param)) {
        const auto& dims = in_dims;
        TransposeVitMiddleAxes(input, output, dims[0], dims[1], dims[2], dims[3]);
        return 0;
    }

    if (IsVitHeadLastTranspose(param)) {
        const auto& dims = in_dims;
        TransposeVitHeadLast(input, output, dims[0], dims[1], dims[2], dims[3]);
        return 0;
    }

    if (Is2DTranspose(param)) {
        const int64_t rows = in_dims[0];
        const int64_t cols = in_dims[1];
        Transpose2DTiled(input, output, rows, cols);
        return 0;
    }

    if (IsLastTwoAxesTranspose(param)) {
        const int64_t rows = in_dims[in_dims.size() - 2];
        const int64_t cols = in_dims.back();
        const int64_t matrix_size = rows * cols;
        const int64_t matrix_count = param->input->numel() / matrix_size;
        for (int64_t matrix = 0; matrix < matrix_count; ++matrix) {
            Transpose2DTiled(input + matrix * matrix_size, output + matrix * matrix_size, rows, cols);
        }
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

template <>
int32_t TransposeKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Transpose::BF16");
    auto* param = static_cast<feather::operators::TransposeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::BF16) {
        return ComputeTransposeFallback<DataType::BF16>(param);
    }
    return ComputeTransposeRaw<BFloat16>(param, DataType::BF16);
}

void EnsureX86TransposeKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
