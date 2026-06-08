#ifndef FEATHER_UTIL_THREADING_H
#define FEATHER_UTIL_THREADING_H

#include <algorithm>
#include <cstdint>
#include <cstddef>

namespace feather {

inline constexpr size_t kDefaultThreadCount = 8;

inline size_t DefaultThreadCount() {
    return kDefaultThreadCount;
}

inline size_t ThreadCountForWorkItems(int64_t total_work_items) {
    if (total_work_items <= 1) {
        return 1;
    }
    return std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(total_work_items), DefaultThreadCount()));
}

}  // namespace feather

#endif  // FEATHER_UTIL_THREADING_H
