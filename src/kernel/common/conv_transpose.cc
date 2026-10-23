// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/kernel/conv_transpose.h"

namespace feather {
namespace kernel {

int32_t CommonConvTransposeKernel::compute() {
    auto* param = ParamAs<operators::ConvTransposeParam>();
    std::vector<int64_t> shape;
    if (!param || !operators::ConvTransposeOutputShape(*param, &shape) ||
        param->out->dims().data() != shape || !param->input->IsInitialized() ||
        !param->weight->IsInitialized() || !param->out->IsInitialized() ||
        (param->bias && !param->bias->IsInitialized())) return -1;
    const auto& input_shape = param->input->dims();
    const auto& weight_shape = param->weight->dims();
    const int64_t batches = input_shape[0], input_channels = input_shape[1];
    const int64_t input_h = input_shape[2], input_w = input_shape[3];
    const int64_t output_channels = shape[1], output_h = shape[2], output_w = shape[3];
    const int64_t kernel_h = weight_shape[2], kernel_w = weight_shape[3];
    const int64_t channels_per_group = input_channels / param->group;
    const int64_t output_per_group = weight_shape[1];
    const auto* input = param->input->data<float>();
    const auto* weight = param->weight->data<float>();
    const auto* bias = param->bias ? param->bias->data<float>() : nullptr;
    auto* output = param->out->mutable_data<float>();
    for (int64_t n = 0; n < batches; ++n) {
        for (int64_t oc = 0; oc < output_channels; ++oc) {
            auto* plane = output + (n * output_channels + oc) * output_h * output_w;
            std::fill(plane, plane + output_h * output_w, bias ? bias[oc] : 0.0f);
        }
        for (int64_t ic = 0; ic < input_channels; ++ic) {
            const int64_t group = ic / channels_per_group;
            const auto* input_plane = input + (n * input_channels + ic) * input_h * input_w;
            for (int64_t oc = 0; oc < output_per_group; ++oc) {
                const auto* filter = weight + (ic * output_per_group + oc) * kernel_h * kernel_w;
                auto* output_plane = output + (n * output_channels + group * output_per_group + oc) * output_h * output_w;
                for (int64_t ih = 0; ih < input_h; ++ih) {
                    for (int64_t iw = 0; iw < input_w; ++iw) {
                        const float value = input_plane[ih * input_w + iw];
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            const __int128 oh = static_cast<__int128>(ih) * param->strides[0] - param->pads[0] +
                                                static_cast<__int128>(kh) * param->dilations[0];
                            if (oh < 0 || oh >= output_h) continue;
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const __int128 ow = static_cast<__int128>(iw) * param->strides[1] - param->pads[1] +
                                                    static_cast<__int128>(kw) * param->dilations[1];
                                if (ow >= 0 && ow < output_w)
                                    output_plane[static_cast<int64_t>(oh) * output_w + static_cast<int64_t>(ow)] +=
                                        value * filter[kh * kernel_w + kw];
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

namespace {
bool registered = [] {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "ConvTranspose",
                                               [] { return std::make_unique<CommonConvTransposeKernel>(); });
    return true;
}();
}  // namespace

void EnsureCommonConvTransposeKernelsRegistered() { (void)registered; }
}  // namespace kernel
}  // namespace feather
