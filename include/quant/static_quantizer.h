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

#ifndef FEATHER_QUANT_STATIC_QUANTIZER_H
#define FEATHER_QUANT_STATIC_QUANTIZER_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "model/model_format.h"

namespace feather {

struct ActivationQuantization {
    float scale{1.0f};
    int32_t zero_point{0};
};

using ActivationQuantizationTable = std::unordered_map<std::string, ActivationQuantization>;

struct StaticQuantizationConfig {
    bool strict{true};
    bool symmetric{true};
    bool per_channel_weights{true};
    // Keep numerically sensitive node subgraphs in their original FP32 form.
    std::vector<std::string> keep_fp32_node_prefixes;
    ActivationQuantizationTable activations;
    ActivationQuantizationTable activation_table;
    ActivationQuantizationTable activation_quantization;
};

struct StaticQuantizationReport {
    std::vector<std::string> quantized_nodes;
    std::vector<std::string> skipped_nodes;
    std::vector<std::string> diagnostics;
};

int32_t LoadActivationQuantizationTable(const std::string& path,
                                        ActivationQuantizationTable* table,
                                        std::vector<std::string>* diagnostics = nullptr);

int32_t StaticQuantizeModel(
    const model::ModelDesc& input_model,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& source_weights,
    const StaticQuantizationConfig& config,
    model::ModelDesc* output_model,
    std::unordered_map<std::string, std::shared_ptr<Tensor>>* output_weights,
    StaticQuantizationReport* report = nullptr);

namespace quant {
using ::feather::ActivationQuantization;
using ::feather::ActivationQuantizationTable;
using ::feather::StaticQuantizationConfig;
using ::feather::StaticQuantizationReport;
using ::feather::LoadActivationQuantizationTable;
using ::feather::StaticQuantizeModel;
}  // namespace quant

}  // namespace feather

#endif  // FEATHER_QUANT_STATIC_QUANTIZER_H
