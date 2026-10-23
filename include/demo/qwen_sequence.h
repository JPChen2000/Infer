// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_QWEN_SEQUENCE_H
#define FEATHER_DEMO_QWEN_SEQUENCE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "model/model_io.h"

namespace feather {
namespace demo {

// Specialize the canonical sequence-annotated graph after context specialization.
// Pass the context-specialized model and its constant overrides. Every explicit
// qwen.sequence binding is validated against the checkpoint's default sequence;
// generated arange constants replace their payload AND tensor descriptor.
// Existing overrides are retained (and merged for shared shape payloads). Outputs
// are committed together on success; failure leaves both arguments unchanged.
// A zero sequence selects the exported default. Legacy fixed graphs accept only
// their own sequence length. Weight names and loader payloads stay unchanged.
bool SpecializeQwenSequence(model::ModelLoader& loader, int64_t sequence,
                            model::ModelDesc* effective_model,
                            std::unordered_map<std::string, std::shared_ptr<Tensor>>* constant_overrides,
                            std::string* error);

}  // namespace demo
}  // namespace feather
#endif
