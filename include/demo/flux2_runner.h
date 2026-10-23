// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_DEMO_FLUX2_RUNNER_H
#define FEATHER_DEMO_FLUX2_RUNNER_H
#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "demo/image_io.h"
#include "demo/qwen_tokenizer.h"
#include "model/model_format.h"
namespace feather {
namespace demo {
enum class Flux2Backend { kHost, kCommon, kX86, kCuda };
bool ParseFlux2Backend(const std::string&, Flux2Backend*);
struct Flux2Options {
    Flux2Backend backend{Flux2Backend::kCommon};
    uint32_t seed{0};
    int steps{4};
};
// Fixed-resolution batch-one pipeline. Calls on one instance must be serialized.
class Flux2Runner {
   public:
    Status Load(const std::string& model_dir, const Flux2Options& options = {});
    Status Generate(const std::string& prompt, ImageData* image);
    Status GenerateFromEmbeddings(const std::vector<float>& embeddings, ImageData* image);
    const std::string& LastError() const { return last_error_; }
    static std::vector<float> Timesteps(int64_t image_sequence, int steps);

   private:
    Status Finish(Status status);
    Status GenerateImpl(const std::vector<float>& embeddings, ImageData* image);
    std::string directory_, last_error_;
    Flux2Options options_;
    QwenTokenizer tokenizer_;
    model::ModelDesc text_, denoiser_, decoder_;
    int64_t height_{0}, width_{0}, length_{0}, context_{0}, channels_{0}, pad_{0};
    bool loaded_{false}, left_padding_{false}, tokenizer_ready_{false};
};
}  // namespace demo
}  // namespace feather
#endif
