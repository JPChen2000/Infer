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

#ifndef FEATHER_CORE_STATUS_H
#define FEATHER_CORE_STATUS_H

#include <string>
#include <utility>

namespace feather {

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kNotFound,
    kShapeMismatch,
    kUnsupported,
    kBuildFailed,
    kExecutionFailed,
    kInternal,
};

class Status {
   public:
    Status() = default;
    explicit Status(StatusCode code) : code_(code) {}
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    static Status Ok() { return Status(); }
    static Status Error(StatusCode code, std::string message) { return Status(code, std::move(message)); }

    Status WithContext(const std::string& context) const {
        return ok() ? *this : Status(code_, context + ": " + message_);
    }
    bool ok() const { return code_ == StatusCode::kOk; }
    explicit operator bool() const { return ok(); }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }

   private:
    StatusCode code_{StatusCode::kOk};
    std::string message_;
};

}  // namespace feather

#endif  // FEATHER_CORE_STATUS_H
