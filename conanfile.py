# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from conans import ConanFile, CMake, tools


class InferConan(ConanFile):
    name = "Infer"
    license = "Apache-2.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "BUILD_UNITTEST": [True, False]}
    default_options = {"shared": False,
                       "fPIC": True,
                       "BUILD_UNITTEST": False}
    generators = "cmake"
    exports_sources = ("CMakeLists.txt", "include*", "src*",
                       "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
    install_folder = "./"

    def requirements(self):
        self.requires("benchmark/1.8.5")
        if self.options.BUILD_UNITTEST:
            self.requires("gtest/1.11.0")

    def configure(self):
        if self.settings.os == "Windows":
            del self.option.fPIC

    def configure_cmake(self):
        cmake = CMake(self)
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        cmake = self.configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
