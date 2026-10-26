// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_INTERNAL_QWEN_MODEL_RESOURCES_H
#define FEATHER_DEMO_INTERNAL_QWEN_MODEL_RESOURCES_H
#include <memory>
#include <string>
#include "model/model_io.h"
namespace feather { namespace demo { namespace internal {
// Built once before publication. Plans specialize temporary loader views and
// never retain duplicate canonical model descriptions or independent mappings.
struct QwenModelSource {
    model::ModelLoader loader;
    std::string path;
};
struct QwenModelResources {
    QwenModelSource decode;
    std::unique_ptr<const QwenModelSource> prefill;
};
} } }
#endif
