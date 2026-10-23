// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_KERNEL_RESIZE_H
#define FEATHER_KERNEL_RESIZE_H

#include <algorithm>
#include <cmath>

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

#if defined(__CUDACC__)
#define FEATHER_RESIZE_HD __host__ __device__
#else
#define FEATHER_RESIZE_HD
#endif

FEATHER_RESIZE_HD inline double ResizeCoordinate(int64_t output, int64_t input_size, int64_t output_size,
                                                 double scale, operators::ResizeParam::CoordinateMode mode) {
    using Mode = operators::ResizeParam::CoordinateMode;
    if (mode == Mode::kAlignCorners) {
        return output_size > 1 ? static_cast<double>(output) * (input_size - 1) / (output_size - 1) : 0.0;
    }
    if (mode == Mode::kPytorchHalfPixel && output_size == 1) return 0.0;
    if (mode == Mode::kHalfPixel || mode == Mode::kPytorchHalfPixel) return (output + 0.5) / scale - 0.5;
    return static_cast<double>(output) / scale;
}

FEATHER_RESIZE_HD inline int64_t ResizeNearestIndex(double coordinate, int64_t size,
                                                   operators::ResizeParam::NearestMode mode) {
    using Mode = operators::ResizeParam::NearestMode;
    // Clamp before conversion, including negative half-pixel border positions.
    if (coordinate <= 0.0) return 0;
    if (coordinate >= static_cast<double>(size - 1)) return size - 1;
    const double lower = floor(coordinate);
    double rounded = lower;
    if (mode == Mode::kCeil) rounded = ceil(coordinate);
    if (mode == Mode::kRoundPreferFloor) rounded = lower + (coordinate - lower > 0.5 ? 1.0 : 0.0);
    if (mode == Mode::kRoundPreferCeil) rounded = lower + (coordinate - lower >= 0.5 ? 1.0 : 0.0);
    return static_cast<int64_t>(rounded);
}

#undef FEATHER_RESIZE_HD

inline double ResizeScale(const operators::ResizeParam& param, size_t axis) {
    return param.effective_scales.size() == param.scales.size() ? param.effective_scales[axis] : param.scales[axis];
}

inline bool IsLegacyResize(const operators::ResizeParam& param) {
    return param.mode == operators::ResizeParam::Mode::kNearest &&
           param.coordinate_mode == operators::ResizeParam::CoordinateMode::kAsymmetric &&
           param.nearest_mode == operators::ResizeParam::NearestMode::kFloor;
}

// Shared host reference: nearest handles every axis; linear is rank-four
// spatial interpolation with FP32 accumulation for both image layouts.
template <typename Read, typename Write>
inline int32_t ComputeResizeHost(operators::ResizeParam* param, Read read, Write write) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.empty() || in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size()) return -1;
    std::vector<int64_t> strides(in_dims.size(), 1);
    for (size_t axis = in_dims.size() - 1; axis > 0; --axis) strides[axis - 1] = strides[axis] * in_dims[axis];
    const bool linear = param->mode == operators::ResizeParam::Mode::kLinear;
    if (linear && in_dims.size() != 4) return -1;
    const int h_axis = HeightAxisForLayout(param->input->layout());
    const int w_axis = WidthAxisForLayout(param->input->layout());
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        int64_t remaining = index;
        int64_t offset = 0;
        int64_t h0 = 0, h1 = 0, w0 = 0, w1 = 0;
        float wh = 0.0f, ww = 0.0f;
        for (int axis = static_cast<int>(out_dims.size()) - 1; axis >= 0; --axis) {
            const int64_t coord = remaining % out_dims[axis];
            remaining /= out_dims[axis];
            const double source = ResizeCoordinate(coord, in_dims[axis], out_dims[axis], ResizeScale(*param, axis),
                                                   param->coordinate_mode);
            if (linear && (axis == h_axis || axis == w_axis)) {
                const double bounded = std::max(0.0, std::min(source, static_cast<double>(in_dims[axis] - 1)));
                const int64_t low = static_cast<int64_t>(floor(bounded));
                const int64_t high = std::min(low + 1, in_dims[axis] - 1);
                if (axis == h_axis) { h0 = low; h1 = high; wh = static_cast<float>(bounded - low); }
                else { w0 = low; w1 = high; ww = static_cast<float>(bounded - low); }
            } else {
                offset += ResizeNearestIndex(source, in_dims[axis], param->nearest_mode) * strides[axis];
            }
        }
        if (!linear) { write(index, read(offset)); continue; }
        const float top = read(offset + h0 * strides[h_axis] + w0 * strides[w_axis]) * (1.0f - ww) +
                          read(offset + h0 * strides[h_axis] + w1 * strides[w_axis]) * ww;
        const float bottom = read(offset + h1 * strides[h_axis] + w0 * strides[w_axis]) * (1.0f - ww) +
                             read(offset + h1 * strides[h_axis] + w1 * strides[w_axis]) * ww;
        write(index, top * (1.0f - wh) + bottom * wh);
    }
    return 0;
}

enum class CudaResizeBackend {
    kUnknown,
    kGeneric,
    kDirect4D,
};

namespace detail {
inline thread_local CudaResizeBackend g_last_cuda_resize_backend = CudaResizeBackend::kUnknown;
}

inline CudaResizeBackend LastCudaResizeBackend() { return detail::g_last_cuda_resize_backend; }
inline void ResetLastCudaResizeBackend() { detail::g_last_cuda_resize_backend = CudaResizeBackend::kUnknown; }
inline void SetLastCudaResizeBackend(CudaResizeBackend backend) { detail::g_last_cuda_resize_backend = backend; }

void EnsureCommonResizeKernelsRegistered();
void EnsureX86ResizeKernelsRegistered();
void EnsureResizeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ResizeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESIZE_H
