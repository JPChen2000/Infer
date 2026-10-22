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

#include <gtest/gtest.h>

#include "core/status.h"

namespace {

TEST(status_test, DefaultsToOk) {
    const feather::Status status;
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.code(), feather::StatusCode::kOk);
    EXPECT_TRUE(status.message().empty());
}

TEST(status_test, PreservesCodeAndDiagnostic) {
    const auto status = feather::Status::Error(feather::StatusCode::kNotFound, "tensor: input");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), feather::StatusCode::kNotFound);
    EXPECT_EQ(status.message(), "tensor: input");
}

}  // namespace
