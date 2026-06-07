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
