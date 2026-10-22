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

#ifndef FEATHER_DEMO_IMAGE_IO_H
#define FEATHER_DEMO_IMAGE_IO_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "demo/detection.h"

namespace feather {
namespace demo {

struct ImageData {
    int width{};
    int height{};
    int channels{};
    std::vector<uint8_t> pixels;
};

struct LetterboxInfo {
    int resized_width{};
    int resized_height{};
    int pad_x{};
    int pad_y{};
    float scale{};
};

int32_t LoadImage(const std::string& path, ImageData* image);

int32_t PreprocessImageToTensor(const ImageData& image, int input_size, DataType dtype,
                                std::shared_ptr<Tensor>* tensor, LetterboxInfo* letterbox);
int32_t PreprocessImageToTensor(const ImageData& image, int input_size, DataType dtype,
                                Tensor* tensor, LetterboxInfo* letterbox);

int32_t SaveDetectionsImage(const ImageData& image, const std::vector<Detection>& detections,
                            const std::string& output_path);

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_IMAGE_IO_H
