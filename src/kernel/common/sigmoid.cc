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

#include "src/kernel/sigmoid.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_sigmoid_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Sigmoid",
                                                []() { return std::make_unique<SigmoidKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Sigmoid",
                                                []() { return std::make_unique<SigmoidKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Sigmoid",
                                                []() { return std::make_unique<SigmoidKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeSigmoidKernel(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        const float input = TensorIO<dtype>::Read(param->input.get(), i);
        TensorIO<dtype>::Write(param->out.get(), i, 1.0f / (1.0f + std::exp(-input)));
    }
    return 0;
}

template <>
int32_t SigmoidKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Sigmoid::FP32");
    return ComputeSigmoidKernel<DataType::FP32>(ParamAs<feather::operators::UnaryParam>());
}

template <>
int32_t SigmoidKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Sigmoid::FP16");
    return ComputeSigmoidKernel<DataType::FP16>(ParamAs<feather::operators::UnaryParam>());
}

template <>
int32_t SigmoidKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Sigmoid::BF16");
    return ComputeSigmoidKernel<DataType::BF16>(ParamAs<feather::operators::UnaryParam>());
}

typedef feather::kernel::SigmoidKernel<DeviceType::COMMON, DataType::FP32> SigmoidCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Sigmoid, SigmoidCommonFP32Kernel);

typedef feather::kernel::SigmoidKernel<DeviceType::COMMON, DataType::FP16> SigmoidCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Sigmoid, SigmoidCommonFP16Kernel);

void EnsureCommonSigmoidKernelsRegistered() { (void)g_sigmoid_kernels_registered; }

void EnsureSigmoidKernelsRegistered() {
    EnsureCommonSigmoidKernelsRegistered();
    EnsureX86SigmoidKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
