#include "demo/image_io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "util/fp16.h"

namespace feather {
namespace demo {

namespace {

constexpr uint8_t kLetterboxValue = 114;
constexpr int kBoxThickness = 2;

void WriteTensorValue(Tensor* tensor, int64_t index, float value) {
    if (tensor->data_type() == DataType::FP16) {
        tensor->mutable_data<uint16_t>()[index] = FloatToHalf(value);
        return;
    }
    tensor->mutable_data<float>()[index] = value;
}

void SetPixel(ImageData* image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (image == nullptr || x < 0 || y < 0 || x >= image->width || y >= image->height) {
        return;
    }
    const size_t offset = static_cast<size_t>((y * image->width + x) * 3);
    image->pixels[offset + 0] = r;
    image->pixels[offset + 1] = g;
    image->pixels[offset + 2] = b;
}

void DrawRectangle(ImageData* image, const Detection& det) {
    const int x1 = std::clamp(static_cast<int>(std::floor(det.x1)), 0, image->width - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(det.y1)), 0, image->height - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(det.x2)), 0, image->width - 1);
    const int y2 = std::clamp(static_cast<int>(std::ceil(det.y2)), 0, image->height - 1);
    if (x2 < x1 || y2 < y1) {
        return;
    }

    const uint8_t r = static_cast<uint8_t>((37 * (det.class_id + 3)) % 255);
    const uint8_t g = static_cast<uint8_t>((17 * (det.class_id + 7)) % 255);
    const uint8_t b = static_cast<uint8_t>((29 * (det.class_id + 11)) % 255);

    for (int t = 0; t < kBoxThickness; ++t) {
        for (int x = x1; x <= x2; ++x) {
            SetPixel(image, x, y1 + t, r, g, b);
            SetPixel(image, x, y2 - t, r, g, b);
        }
        for (int y = y1; y <= y2; ++y) {
            SetPixel(image, x1 + t, y, r, g, b);
            SetPixel(image, x2 - t, y, r, g, b);
        }
    }
}

}  // namespace

int32_t LoadImage(const std::string& path, ImageData* image) {
    if (image == nullptr) {
        return -1;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr || width <= 0 || height <= 0) {
        return -1;
    }

    image->width = width;
    image->height = height;
    image->channels = 3;
    image->pixels.assign(raw, raw + static_cast<size_t>(width * height * 3));
    stbi_image_free(raw);
    return 0;
}

int32_t PreprocessImageToTensor(const ImageData& image, int input_size, DataType dtype,
                                Tensor* tensor, LetterboxInfo* letterbox) {
    if (tensor == nullptr || letterbox == nullptr || image.width <= 0 || image.height <= 0 ||
        image.channels != 3 || input_size <= 0) {
        return -1;
    }
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        return -1;
    }

    const float scale =
        std::min(static_cast<float>(input_size) / static_cast<float>(image.width),
                 static_cast<float>(input_size) / static_cast<float>(image.height));
    const int resized_width = std::max(1, static_cast<int>(std::round(image.width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(image.height * scale)));
    const int pad_x = (input_size - resized_width) / 2;
    const int pad_y = (input_size - resized_height) / 2;

    std::vector<uint8_t> resized(static_cast<size_t>(resized_width * resized_height * 3), 0);
    for (int y = 0; y < resized_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) / scale - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
        const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
        const float ly = src_y - std::floor(src_y);
        for (int x = 0; x < resized_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
            const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
            const float lx = src_x - std::floor(src_x);
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int px, int py) -> float {
                    return static_cast<float>(image.pixels[(py * image.width + px) * 3 + c]);
                };
                const float top = at(x0, y0) * (1.0f - lx) + at(x1, y0) * lx;
                const float bottom = at(x0, y1) * (1.0f - lx) + at(x1, y1) * lx;
                const float value = top * (1.0f - ly) + bottom * ly;
                resized[(y * resized_width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
            }
        }
    }

    std::vector<uint8_t> canvas(static_cast<size_t>(input_size * input_size * 3), kLetterboxValue);
    for (int y = 0; y < resized_height; ++y) {
        for (int x = 0; x < resized_width; ++x) {
            const size_t src_offset = static_cast<size_t>((y * resized_width + x) * 3);
            const size_t dst_offset = static_cast<size_t>(((y + pad_y) * input_size + (x + pad_x)) * 3);
            std::memcpy(canvas.data() + dst_offset, resized.data() + src_offset, 3);
        }
    }

    tensor->Resize(std::vector<int64_t>{1, 3, input_size, input_size});
    tensor->set_data_type(dtype);
    if (dtype == DataType::FP16) {
        (void)tensor->mutable_data<uint16_t>();
    } else {
        (void)tensor->mutable_data<float>();
    }

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_size; ++y) {
            for (int x = 0; x < input_size; ++x) {
                const size_t src_index = static_cast<size_t>((y * input_size + x) * 3 + c);
                const int64_t dst_index =
                    static_cast<int64_t>(c) * input_size * input_size + y * input_size + x;
                WriteTensorValue(tensor, dst_index, static_cast<float>(canvas[src_index]) / 255.0f);
            }
        }
    }

    letterbox->resized_width = resized_width;
    letterbox->resized_height = resized_height;
    letterbox->pad_x = pad_x;
    letterbox->pad_y = pad_y;
    letterbox->scale = scale;
    return 0;
}

int32_t PreprocessImageToTensor(const ImageData& image, int input_size, DataType dtype,
                                std::shared_ptr<Tensor>* tensor, LetterboxInfo* letterbox) {
    if (tensor == nullptr) {
        return -1;
    }
    if (*tensor == nullptr) {
        *tensor = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, input_size, input_size});
    }
    return PreprocessImageToTensor(image, input_size, dtype, tensor->get(), letterbox);
}

int32_t SaveDetectionsImage(const ImageData& image, const std::vector<Detection>& detections,
                            const std::string& output_path) {
    if (image.width <= 0 || image.height <= 0 || image.channels != 3 || image.pixels.empty() ||
        output_path.empty()) {
        return -1;
    }

    ImageData annotated = image;
    for (const auto& det : detections) {
        DrawRectangle(&annotated, det);
    }

    const int ok = stbi_write_jpg(output_path.c_str(), annotated.width, annotated.height, 3,
                                  annotated.pixels.data(), 95);
    return ok == 0 ? -1 : 0;
}

}  // namespace demo
}  // namespace feather
