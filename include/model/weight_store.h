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

#ifndef FEATHER_MODEL_WEIGHT_STORE_H
#define FEATHER_MODEL_WEIGHT_STORE_H

#include <memory>
#include <string>
#include <unordered_map>

#include "core/memory.h"
#include "core/tensor.h"
#include "model/model_format.h"

namespace feather {
namespace model {

class MappedFile {
   public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool Open(const std::string& path);
    void Close();

    void* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }

   private:
    int fd_{-1};
    void* data_{};
    size_t size_{};
    std::string path_;
};

class WeightStore {
   public:
    bool AddShard(const std::string& path);
    std::shared_ptr<Tensor> CreateTensorView(const TensorDesc& desc, const WeightLocation& location);

   private:
    std::shared_ptr<MappedFile> GetOrOpenShard(const std::string& path);

    std::unordered_map<std::string, std::shared_ptr<MappedFile>> shards_;
};

}  // namespace model
}  // namespace feather

#endif  // FEATHER_MODEL_WEIGHT_STORE_H
