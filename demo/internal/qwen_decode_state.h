// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_INTERNAL_QWEN_DECODE_STATE_H
#define FEATHER_DEMO_INTERNAL_QWEN_DECODE_STATE_H
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include "core/tensor.h"
namespace feather { namespace demo { namespace internal {
// One allocation per public request. Every shape plan binds these same Tensor
// objects; sharing model resources never shares this mutable state.
struct QwenDecodeState {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    int64_t tokens_processed{0};
    uint64_t revision{0};
    bool invalid{false};
};
} } }
#endif
