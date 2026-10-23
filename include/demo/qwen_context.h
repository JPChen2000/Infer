// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_QWEN_CONTEXT_H
#define FEATHER_DEMO_QWEN_CONTEXT_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "model/model_io.h"

namespace feather {
namespace demo {

// Resolve explicitly annotated context expressions before graph compilation.
// Outputs are committed only after every binding has been validated.
bool SpecializeQwenContext(model::ModelLoader& loader, int64_t requested_context,
                           model::ModelDesc* effective_model,
                           std::unordered_map<std::string, std::shared_ptr<Tensor>>* constant_overrides,
                           int64_t* model_context_limit, std::string* error);

}  // namespace demo
}  // namespace feather
#endif
