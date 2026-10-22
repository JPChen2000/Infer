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

#ifndef FEATHER_TEST_PYTHON_TEST_UTILS_H
#define FEATHER_TEST_PYTHON_TEST_UTILS_H

#include <cstdlib>
#include <filesystem>
#include <string>

namespace feather { namespace test {
inline std::string ShellQuote(const std::string& value) {
    std::string result = "'";
    for (const char ch : value) result += ch == '\'' ? "'\\''" : std::string(1, ch);
    return result + "'";
}
inline std::string PythonCommand() {
    const auto* python = std::getenv("FEATHER_TEST_PYTHON");
    return ShellQuote(python && *python ? python : "python3");
}
inline bool HasPythonModules(const std::string& modules) {
    const auto command = PythonCommand() + " -c " + ShellQuote("import " + modules) + " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
}
inline std::filesystem::path OptionalModelPath(const char* variable, const std::filesystem::path& fallback) {
    const auto* path = std::getenv(variable);
    return path && *path ? std::filesystem::path(path) : fallback;
}
} }  // namespace feather::test
#endif
