#ifndef FEATHER_KERNEL_X86_PREPARED_WEIGHT_CACHE_H
#define FEATHER_KERNEL_X86_PREPARED_WEIGHT_CACHE_H

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "core/tensor.h"

namespace feather { namespace kernel { namespace x86 { namespace detail {

inline bool ValidPackedSource(const Tensor& tensor, int64_t k, int64_t n, bool fp8) {
    const auto dtype = tensor.data_type();
    if (!tensor.IsInitialized() || k <= 0 || n <= 0 ||
        (fp8 ? (dtype != DataType::FP8E4M3 && dtype != DataType::FP8E5M2) : dtype != DataType::BF16)) return false;
    size_t bytes = 0;
    return CheckedByteCount({k, n}, fp8 ? 1 : 2, &bytes) &&
           tensor.offset() <= tensor.storage()->size() &&
           bytes <= tensor.storage()->size() - tensor.offset();
}

// Only physically readonly sources may share preparations. Each value retains
// its backing buffer, preventing pointer reuse while a packed owner is alive.
// Weak entries release model data when the last kernel/request releases it.
struct PreparedWeight {
    std::shared_ptr<const Buffer> source;
    std::shared_ptr<const std::vector<uint16_t>> values;
};

struct PreparedWeightKey {
    const void* source;
    DataType dtype;
    int64_t k, n;
    bool transposed;
    bool operator<(const PreparedWeightKey& other) const {
        if (source != other.source) return std::less<const void*>{}(source, other.source);
        return std::tie(dtype, k, n, transposed) < std::tie(other.dtype, other.k, other.n, other.transposed);
    }
};

struct PreparedWeightCache {
    std::mutex mutex;
    std::map<PreparedWeightKey, std::weak_ptr<const PreparedWeight>> entries;
    size_t acquisitions{0};
};
inline PreparedWeightCache& WeightCache() {
    static PreparedWeightCache cache;
    return cache;
}

template<class Factory>
std::shared_ptr<const std::vector<uint16_t>> AcquirePackedWeight(
    const Tensor& tensor, int64_t k, int64_t n, bool transposed, Factory prepare) {
    auto& cache = WeightCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    try {
        if (++cache.acquisitions % 128 == 0) {
            for (auto it = cache.entries.begin(); it != cache.entries.end();) {
                if (it->second.expired()) it = cache.entries.erase(it);
                else ++it;
            }
        }
        const PreparedWeightKey key{tensor.raw_data(), tensor.data_type(), k, n, transposed};
        auto& entry = cache.entries[key];
        auto packed = entry.lock();
        if (!packed) {
            auto values = prepare();
            if (!values) return {};
            packed = std::make_shared<PreparedWeight>(PreparedWeight{tensor.storage()->buffer(), std::move(values)});
            entry = packed;
        }
        const auto* values = packed->values.get();
        return std::shared_ptr<const std::vector<uint16_t>>(std::move(packed), values);
    } catch (const std::bad_alloc&) {
        return {};
    }
}

}}}} // namespace feather::kernel::x86::detail
#endif
