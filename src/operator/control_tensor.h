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

#ifndef FEATHER_OPERATOR_CONTROL_TENSOR_H
#define FEATHER_OPERATOR_CONTROL_TENSOR_H

#include <cstdint>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "util/bf16.h"
#include "util/fp16.h"

namespace feather {
namespace operators {

inline bool ReadIntegerTensor(const std::shared_ptr<Tensor>& tensor, std::vector<int64_t>* values) {
    if (tensor == nullptr || values == nullptr || !tensor->IsInitialized() || tensor->numel() < 0) {
        return false;
    }

    values->clear();
    values->reserve(static_cast<size_t>(tensor->numel()));
    for (int64_t index = 0; index < tensor->numel(); ++index) {
        switch (tensor->data_type()) {
            case DataType::BOOL:
            case DataType::UINT8:
                values->push_back(tensor->data<uint8_t>()[index]);
                break;
            case DataType::INT8:
                values->push_back(tensor->data<int8_t>()[index]);
                break;
            case DataType::INT32:
                values->push_back(tensor->data<int32_t>()[index]);
                break;
            case DataType::INT64:
                values->push_back(tensor->data<int64_t>()[index]);
                break;
            default:
                return false;
        }
    }
    return true;
}

inline bool ReadFloatTensor(const std::shared_ptr<Tensor>& tensor, std::vector<float>* values) {
    if (tensor == nullptr || values == nullptr || !tensor->IsInitialized() || tensor->numel() < 0) {
        return false;
    }

    values->clear();
    values->reserve(static_cast<size_t>(tensor->numel()));
    for (int64_t index = 0; index < tensor->numel(); ++index) {
        switch (tensor->data_type()) {
            case DataType::FP16:
                values->push_back(HalfToFloat(tensor->data<uint16_t>()[index]));
                break;
            case DataType::BF16:
                values->push_back(BFloat16ToFloat(tensor->data<BFloat16>()[index].bits));
                break;
            case DataType::FP32:
                values->push_back(tensor->data<float>()[index]);
                break;
            case DataType::INT32:
                values->push_back(static_cast<float>(tensor->data<int32_t>()[index]));
                break;
            case DataType::INT64:
                values->push_back(static_cast<float>(tensor->data<int64_t>()[index]));
                break;
            default:
                return false;
        }
    }
    return true;
}

inline bool ReadScalarFloatTensor(const std::shared_ptr<Tensor>& tensor, float* value) {
    if (value == nullptr) {
        return false;
    }
    std::vector<float> values;
    if (!ReadFloatTensor(tensor, &values) || values.size() != 1) {
        return false;
    }
    *value = values.front();
    return true;
}

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_CONTROL_TENSOR_H
