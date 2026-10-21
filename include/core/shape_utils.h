#ifndef FEATHER_CORE_SHAPE_UTILS_H
#define FEATHER_CORE_SHAPE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "util/types.h"

namespace feather {

inline bool CheckedElementCount(const std::vector<int64_t>& dims, size_t* count) {
    if (count == nullptr) return false;
    bool empty = false;
    for (const auto dim : dims) {
        if (dim < 0) return false;
        empty = empty || dim == 0;
    }
    size_t result = empty ? 0 : 1;
    for (const auto dim : dims) {
        if (dim != 0 && result > static_cast<size_t>(std::numeric_limits<int64_t>::max()) /
                                        static_cast<uint64_t>(dim)) return false;
        result *= static_cast<size_t>(dim);
    }
    *count = result;
    return true;
}

inline bool CheckedByteCount(const std::vector<int64_t>& dims, size_t element_bytes, size_t* bytes) {
    size_t count = 0;
    if (bytes == nullptr || element_bytes == 0 || !CheckedElementCount(dims, &count) ||
        count > std::numeric_limits<size_t>::max() / element_bytes) return false;
    *bytes = count * element_bytes;
    return true;
}

inline bool CheckedTensorBytes(const std::vector<int64_t>& dims, DataType dtype, size_t* bytes) {
    if (dtype == DataType::INT4) {
        size_t count = 0;
        if (bytes == nullptr || !CheckedElementCount(dims, &count)) return false;
        *bytes = count / 2 + count % 2;
        return true;
    }
    return CheckedByteCount(dims, DataTypeBytes(dtype), bytes);
}

}  // namespace feather
#endif
