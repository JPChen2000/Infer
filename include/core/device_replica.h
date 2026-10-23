// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_CORE_DEVICE_REPLICA_H
#define FEATHER_CORE_DEVICE_REPLICA_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace feather {
namespace detail {

// Allocation/alias-group identity is independent of both host pointer movement
// and mutation versions. A replica represents exactly one byte range.
struct DeviceViewIdentity {
    uint64_t storage_id{0};
    size_t offset{0};
    size_t bytes{0};
    bool operator==(const DeviceViewIdentity& other) const {
        return storage_id == other.storage_id && offset == other.offset && bytes == other.bytes;
    }
    bool operator!=(const DeviceViewIdentity& other) const { return !(*this == other); }
    bool Overlaps(const DeviceViewIdentity& other) const {
        return storage_id == other.storage_id && bytes && other.bytes &&
            (offset <= other.offset ? other.offset - offset < bytes : offset - other.offset < other.bytes);
    }
    bool Contains(const DeviceViewIdentity& other) const {
        return storage_id == other.storage_id && other.offset >= offset &&
            other.offset - offset <= bytes && other.bytes <= bytes - (other.offset - offset);
    }
};

struct DeviceViewIdentityHash {
    size_t operator()(const DeviceViewIdentity& value) const {
        size_t result = std::hash<uint64_t>{}(value.storage_id);
        const auto combine = [&](size_t part) { result ^= part + 0x9e3779b9 + (result << 6) + (result >> 2); };
        combine(std::hash<size_t>{}(value.offset));
        combine(std::hash<size_t>{}(value.bytes));
        return result;
    }
};

// Backend-private cache primitive. The primary table and storage-local alias
// index contain stable view identities. Opaque handles only register users of
// those views and can be detached without dereferencing their former objects.
template <typename State>
class DeviceReplicaIndex {
 public:
    using Views = std::unordered_set<DeviceViewIdentity, DeviceViewIdentityHash>;
    using Map = std::unordered_map<DeviceViewIdentity, State, DeviceViewIdentityHash>;
    using iterator = typename Map::iterator;
    using const_iterator = typename Map::const_iterator;

    template <typename OnOrphan>
    State& Bind(const void* handle, DeviceViewIdentity view, OnOrphan on_orphan) {
        const auto previous = handles_.find(handle);
        if (previous != handles_.end() && previous->second != view) EraseHandle(handle, on_orphan);
        auto inserted = replicas_.try_emplace(view);
        if (inserted.second) storage_views_[view.storage_id].insert(view);
        handles_[handle] = view;
        view_handles_[view].insert(handle);
        return inserted.first->second;
    }

    template <typename OnOrphan>
    void EraseHandle(const void* handle, OnOrphan on_orphan) {
        const auto found = handles_.find(handle);
        if (found == handles_.end()) return;
        const auto view = found->second;
        auto& users = view_handles_.at(view);
        users.erase(handle);
        handles_.erase(found);
        if (!users.empty()) return;
        const auto state = replicas_.find(view);
        on_orphan(state->second);
        replicas_.erase(state);
        view_handles_.erase(view);
        auto aliases = storage_views_.find(view.storage_id);
        aliases->second.erase(view);
        if (aliases->second.empty()) storage_views_.erase(aliases);
    }

    iterator FindHandle(const void* handle) {
        const auto found = handles_.find(handle);
        return found == handles_.end() ? end() : find(found->second);
    }
    const_iterator FindHandle(const void* handle) const {
        const auto found = handles_.find(handle);
        return found == handles_.end() ? end() : find(found->second);
    }
    const Views& StorageViews(uint64_t storage_id) const {
        const auto found = storage_views_.find(storage_id);
        static const Views empty;
        return found == storage_views_.end() ? empty : found->second;
    }
    const std::unordered_set<const void*>& Handles(DeviceViewIdentity view) const { return view_handles_.at(view); }
    iterator find(DeviceViewIdentity view) { return replicas_.find(view); }
    const_iterator find(DeviceViewIdentity view) const { return replicas_.find(view); }
    iterator begin() { return replicas_.begin(); }
    const_iterator begin() const { return replicas_.begin(); }
    iterator end() { return replicas_.end(); }
    const_iterator end() const { return replicas_.end(); }
    size_t size() const { return replicas_.size(); }
    void clear() { replicas_.clear(); storage_views_.clear(); handles_.clear(); view_handles_.clear(); }
    void swap(DeviceReplicaIndex& other) {
        replicas_.swap(other.replicas_); storage_views_.swap(other.storage_views_);
        handles_.swap(other.handles_); view_handles_.swap(other.view_handles_);
    }

 private:
    Map replicas_;
    std::unordered_map<uint64_t, Views> storage_views_;
    std::unordered_map<const void*, DeviceViewIdentity> handles_;
    std::unordered_map<DeviceViewIdentity, std::unordered_set<const void*>, DeviceViewIdentityHash> view_handles_;
};

}  // namespace detail
}  // namespace feather
#endif
