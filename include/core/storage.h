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

#ifndef FEATHER_CORE_STORAGE_H
#define FEATHER_CORE_STORAGE_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include "core/memory.h"

namespace feather {

namespace kernel { namespace cuda_detail { class StorageAccess; } }

// Shared allocation identity. Tensor supplies shape and view offset; Storage
// supplies ownership and mutation tracking for all views of that allocation.
class DeferredAllocationScope {
 public:
    DeferredAllocationScope() : previous_(enabled_) { enabled_ = true; }
    ~DeferredAllocationScope() { enabled_ = previous_; }
    static bool Enabled() { return enabled_; }
 private:
    bool previous_;
    inline static thread_local bool enabled_{false};
};

class Storage {
 public:
    explicit Storage(std::shared_ptr<const Buffer> buffer, DeviceType device = DeviceType::COMMON, int device_index = 0)
        : buffer_(std::move(buffer)), capacity_(buffer_ ? buffer_->size() : 0),
          device_(device), device_index_(device_index), materialized_(true) { MarkMutated(); }
    explicit Storage(size_t bytes) : capacity_(bytes), deferred_(true) { MarkMutated(); }
    const void* data() const {
        if (deferred_) std::call_once(allocation_once_, [this] {
            buffer_ = std::make_shared<Buffer>(capacity_);
            materialized_.store(true, std::memory_order_release);
        });
        return buffer_ ? buffer_->data() : nullptr;
    }
    bool available() const { return deferred_ || (buffer_ && buffer_->data()); }
    size_t size() const { return capacity_; }
    bool read_only() const { return materialized_.load(std::memory_order_acquire) && buffer_ && buffer_->read_only(); }
    DeviceType device() const { return device_; }
    int device_index() const { return device_index_; }
    bool materialized() const { return materialized_.load(std::memory_order_acquire); }
    std::shared_ptr<const Buffer> buffer() const { (void)data(); return buffer_; }
    uint64_t id() const { return id_; }
    uint64_t version() const { return version_.load(std::memory_order_relaxed); }

 private:
    friend class Tensor;
    // Defined only in the CUDA implementation for backend-owned allocations.
    // Tensor host writes still go through its tracked copy-on-write boundary.
    friend class kernel::cuda_detail::StorageAccess;
    void* MutableData() const { (void)data(); return buffer_ ? buffer_->MutableData() : nullptr; }
    void MakeWritable() const {
        (void)data();
        std::lock_guard<std::mutex> lock(writable_mutex_);
        if (!buffer_ || !buffer_->read_only()) return;
        auto copy = std::make_shared<Buffer>(capacity_);
        if (capacity_) std::memcpy(copy->MutableData(), buffer_->data(), capacity_);
        buffer_ = std::move(copy);
        MarkMutated();
    }
    void MarkMutated() const {
        static std::atomic<uint64_t> next{1};
        version_.store(next.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
    }
    static uint64_t NextIdentity() {
        static std::atomic<uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }
    const uint64_t id_{NextIdentity()};
    mutable std::shared_ptr<const Buffer> buffer_;
    size_t capacity_{0};
    bool deferred_{false};
    DeviceType device_{DeviceType::COMMON};
    int device_index_{0};
    mutable std::atomic<bool> materialized_{false};
    mutable std::once_flag allocation_once_;
    mutable std::atomic<uint64_t> version_{0};
    mutable std::mutex writable_mutex_;
};

}  // namespace feather
#endif
