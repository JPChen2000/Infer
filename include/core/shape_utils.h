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
