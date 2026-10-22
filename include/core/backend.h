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

#ifndef FEATHER_CORE_BACKEND_H
#define FEATHER_CORE_BACKEND_H

#include <memory>
#include <functional>
#include "core/status.h"
#include "core/tensor.h"

namespace feather {
struct BackendCapabilities {
    bool supported{false};
    bool host_memory{false};
    bool parallel_nodes{false};
    bool activation_reuse{false};
};
#ifdef FEATHER_WITH_CUDA
namespace kernel { namespace cuda_detail { class CudaExecutionContext; } }
#endif

class ExecutionCapture {
 public:
    virtual ~ExecutionCapture() = default;
    virtual Status Launch() = 0;
};

// The executor knows transfer/coherence operations, not a vendor runtime API.
class BackendContext {
 public:
    class Scope { public: virtual ~Scope() = default; };
    virtual ~BackendContext() = default;
    virtual BackendCapabilities Capabilities(DeviceType device) const {
        switch (device) {
            case DeviceType::UNKNOWN:  // Custom, manually constructed host kernels.
            case DeviceType::COMMON:
            case DeviceType::X86:
            case DeviceType::ARM32:
            case DeviceType::ARM64: return {true, true, true, true};
            default: return {};
        }
    }
    virtual std::unique_ptr<Scope> Enter() { return nullptr; }
    virtual Status Capture(const std::function<Status()>&, std::unique_ptr<ExecutionCapture>*) {
        return Status::Error(StatusCode::kUnsupported, "execution capture unavailable");
    }
    virtual Status Synchronize() { return Status::Ok(); }
    virtual void SetWorkspaceLimit(size_t) {}
    virtual Status ReadHost(Tensor&) { return Status::Ok(); }
    virtual void HostWritten(Tensor&) {}
    virtual void MarkPersistent(const Tensor&, bool) {}
    virtual bool IsPersistent(const Tensor&) const { return false; }
    virtual void Release(const Tensor&) {}
    virtual std::shared_ptr<const Storage> DeviceStorage(const Tensor&) const { return nullptr; }
    virtual void Clear() {}
    // Return Unsupported so model-independent state code can use the host path.
    virtual Status SwapState(Tensor&, Tensor&) { return Status::Error(StatusCode::kUnsupported, "device state swap unavailable"); }
    virtual Status AppendState(Tensor&, const Tensor&, int) { return Status::Error(StatusCode::kUnsupported, "device state append unavailable"); }
#ifdef FEATHER_WITH_CUDA
    virtual const std::shared_ptr<kernel::cuda_detail::CudaExecutionContext>& CudaContext() const {
        static const std::shared_ptr<kernel::cuda_detail::CudaExecutionContext> empty;
        return empty;
    }
#endif
};

std::unique_ptr<BackendContext> CreateBackendContext();

}  // namespace feather
#endif
