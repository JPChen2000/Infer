// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <limits>
#include <stdexcept>

#include "demo/flux2_runner.h"
#include "third_party/stb/stb_image_write.h"
int main(int argc, char** argv) {
    feather::demo::Flux2Options options;
    std::string dir, prompt, output = "flux2.png";
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            std::string value = argv[++i];
            if (arg == "--model-dir")
                dir = value;
            else if (arg == "--prompt")
                prompt = value;
            else if (arg == "--output")
                output = value;
            else if (arg == "--backend") {
                if (!feather::demo::ParseFlux2Backend(value, &options.backend))
                    throw std::runtime_error("invalid backend");
            } else if (arg == "--seed" || arg == "--steps") {
                size_t end = 0;
                if (value.empty() || value[0] == '-') throw std::runtime_error("invalid " + arg);
                auto n = std::stoull(value, &end);
                if (end != value.size() || n > (arg == "--seed" ? std::numeric_limits<uint32_t>::max() : 10000u) ||
                    (arg == "--steps" && n == 0))
                    throw std::runtime_error("invalid " + arg);
                if (arg == "--seed")
                    options.seed = static_cast<uint32_t>(n);
                else
                    options.steps = static_cast<int>(n);
            } else
                throw std::runtime_error("unknown option " + arg);
        }
        if (dir.empty())
            throw std::runtime_error(
                "usage: flux2_infer_demo --model-dir DIR --prompt TEXT [--backend common|host|x86|cuda] [--seed N] "
                "[--steps N] [--output IMAGE.png]");
        feather::demo::Flux2Runner runner;
        auto status = runner.Load(dir, options);
        feather::demo::ImageData image;
        if (status.ok()) status = runner.Generate(prompt, &image);
        if (!status.ok()) throw std::runtime_error(status.message());
        if (!stbi_write_png(output.c_str(), image.width, image.height, 3, image.pixels.data(), image.width * 3))
            throw std::runtime_error("failed to write " + output);
        std::cout << "Saved " << output << " (" << image.width << "x" << image.height << ")\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
