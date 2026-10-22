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

#ifndef FEATHER_UTIL_THREADING_H
#define FEATHER_UTIL_THREADING_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace feather {

inline constexpr size_t kDefaultThreadCount = 8;

inline size_t DefaultThreadCount() {
    const char* configured = std::getenv("FEATHER_NUM_THREADS");
    if (configured != nullptr && configured[0] != '\0') {
        char* end = nullptr;
        const long value = std::strtol(configured, &end, 10);
        if (end != configured && *end == '\0' && value > 0) {
            return static_cast<size_t>(value);
        }
    }
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
