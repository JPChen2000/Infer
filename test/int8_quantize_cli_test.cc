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

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quant/static_quantizer.h"

namespace {

TEST(int8_quantize_cli_test, ScaleTableRejectsExtraColumns) {
    const std::string path = "/tmp/int8_quantize_cli_extra_columns.txt";
    std::ofstream file(path);
    file << "input 0.5 0 extra\n";
    file.close();
    feather::ActivationQuantizationTable table;
    std::vector<std::string> diagnostics;
    EXPECT_EQ(feather::LoadActivationQuantizationTable(path, &table, &diagnostics), -1);
    EXPECT_FALSE(diagnostics.empty());
}

}  // namespace
