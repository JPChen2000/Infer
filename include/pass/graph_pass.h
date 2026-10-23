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

#ifndef FEATHER_PASS_GRAPH_PASS_H
#define FEATHER_PASS_GRAPH_PASS_H

#include <memory>
#include <string>
#include <vector>
#include "core/status.h"
#include "util/types.h"

namespace feather {

class StaticGraph;

class GraphPass {
   public:
    virtual ~GraphPass() = default;
    virtual const std::string& name() const = 0;
    virtual int32_t Run(StaticGraph* graph) = 0;
    virtual std::vector<std::string> Requires() const { return {}; }
    virtual bool SupportsDevice(DeviceType) const { return true; }
};

class PassManager {
   public:
    void AddPass(std::unique_ptr<GraphPass> pass);
    int32_t Run(StaticGraph* graph) const;
    Status RunStatus(StaticGraph* graph) const;
    size_t PassCount() const { return passes_.size(); }

   private:
    std::vector<std::unique_ptr<GraphPass>> passes_;
};

std::shared_ptr<PassManager> CreateDefaultPassManager();
std::shared_ptr<PassManager> CreateYoloPassManager();
std::shared_ptr<PassManager> CreateQwenPassManager(bool fuse_logits_argmax = true);

}  // namespace feather

#endif  // FEATHER_PASS_GRAPH_PASS_H
