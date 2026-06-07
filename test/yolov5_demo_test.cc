#include <gtest/gtest.h>

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "demo/image_io.h"
#include "demo/yolov5_postprocess.h"
#include "demo/yolov5_runner.h"
#include "model/model_io.h"

namespace {

std::filesystem::path WriteTestPpmImage(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "P6\n2 2\n255\n";
    const std::vector<unsigned char> pixels = {
        255, 0,   0,   0,   255, 0,
        0,   0, 255, 255, 255, 255,
    };
    out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    return path;
}

}  // namespace

TEST(yolov5_demo_test, DecodeDetectionsAppliesConfidenceAndNms) {
    feather::Tensor output;
    output.Assign<float>({
        320.f, 320.f, 100.f, 120.f, 0.9f, 0.1f, 0.8f,
        322.f, 318.f, 100.f, 120.f, 0.88f, 0.1f, 0.79f,
        100.f, 100.f, 40.f,  30.f,  0.2f, 0.9f, 0.1f,
    }, {1, 3, 7});

    feather::demo::LetterboxInfo letterbox;
    letterbox.resized_width = 640;
    letterbox.resized_height = 640;
    letterbox.pad_x = 0;
    letterbox.pad_y = 0;
    letterbox.scale = 1.0f;

    const auto detections = feather::demo::DecodeYolov5Detections(output, letterbox, 640, 640, 0.25f, 0.45f);
    ASSERT_EQ(detections.size(), 1U);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_GT(detections[0].score, 0.7f);
}

TEST(yolov5_demo_test, PreprocessImageBuildsNchwTensor) {
    feather::demo::ImageData image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0,   0,   0,   255, 0,
        0,   0, 255, 255, 255, 255,
    };

    std::shared_ptr<feather::Tensor> tensor;
    feather::demo::LetterboxInfo letterbox;
    ASSERT_EQ(feather::demo::PreprocessImageToTensor(image, 4, feather::DataType::FP32, &tensor, &letterbox), 0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims().data(), std::vector<int64_t>({1, 3, 4, 4}));
    EXPECT_EQ(letterbox.resized_width, 4);
    EXPECT_EQ(letterbox.resized_height, 4);
    EXPECT_EQ(letterbox.pad_x, 0);
    EXPECT_EQ(letterbox.pad_y, 0);
    EXPECT_NEAR(letterbox.scale, 2.0f, 1e-6f);
    EXPECT_GE(tensor->data<float>()[0], 0.0f);
    EXPECT_LE(tensor->data<float>()[0], 1.0f);
    EXPECT_NEAR(tensor->data<float>()[0], 159.0f / 255.0f, 1e-5f);
    EXPECT_TRUE(std::isfinite(tensor->data<float>()[15]));
    EXPECT_TRUE(std::isfinite(tensor->data<float>()[31]));
    EXPECT_TRUE(std::isfinite(tensor->data<float>()[47]));
}

TEST(yolov5_demo_test, PreprocessImageCanWriteIntoExistingTensor) {
    feather::demo::ImageData image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0,   0,   0,   255, 0,
        0,   0, 255, 255, 255, 255,
    };

    auto tensor = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 3, 4, 4});
    tensor->set_data_type(feather::DataType::FP16);
    const auto* original_buffer = tensor->raw_data();
    (void)tensor->mutable_data<uint16_t>();

    feather::demo::LetterboxInfo letterbox;
    ASSERT_EQ(feather::demo::PreprocessImageToTensor(image, 4, feather::DataType::FP16, tensor.get(), &letterbox), 0);
    EXPECT_EQ(tensor->raw_data(), original_buffer);
    EXPECT_EQ(tensor->dims().data(), std::vector<int64_t>({1, 3, 4, 4}));
    EXPECT_EQ(tensor->data_type(), feather::DataType::FP16);
    EXPECT_EQ(letterbox.resized_width, 4);
    EXPECT_EQ(letterbox.resized_height, 4);
    EXPECT_EQ(letterbox.pad_x, 0);
    EXPECT_EQ(letterbox.pad_y, 0);
    EXPECT_NEAR(letterbox.scale, 2.0f, 1e-6f);
    EXPECT_NE(tensor->data<uint16_t>()[0], 0);
}

TEST(yolov5_demo_test, LoadFthAndRunImageInference) {
    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto script_path = repo_root / "tools" / "onnx_to_feather.py";
    const auto onnx_path = repo_root / "third_party" / "models" / "yolov5n.onnx";
    const auto output_path = std::filesystem::temp_directory_path() / "yolov5n_demo_test.fth";
    const auto image_path = WriteTestPpmImage(std::filesystem::temp_directory_path() / "yolov5n_demo_test.ppm");

    ASSERT_TRUE(std::filesystem::exists(onnx_path));
    const std::string command = "/home/jarvis/miniconda3/bin/python3 \"" + script_path.string() + "\" --input \"" +
                                onnx_path.string() + "\" --output \"" + output_path.string() + "\"";
    ASSERT_EQ(std::system(command.c_str()), 0);

    feather::demo::Yolov5Runner runner;
    ASSERT_EQ(runner.Load(output_path.string()), 0);
    EXPECT_FALSE(runner.DescribeLastBuild().empty());
    EXPECT_NE(runner.DescribeLastBuild().find("static_nodes="), std::string::npos);
    EXPECT_NE(runner.DescribeLastBuild().find("runtime_nodes="), std::string::npos);

    std::vector<feather::demo::Detection> detections;
    ASSERT_EQ(runner.Run(image_path.string(), 0.25f, 0.45f, &detections), 0);
    EXPECT_NE(runner.DescribeLastRun().find("load_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("preprocess_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("build_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("lower_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("rungraph_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("postprocess_ms="), std::string::npos);
}

TEST(yolov5_demo_test, SaveDetectionsImageWritesAnnotatedOutput) {
    const auto input_path = WriteTestPpmImage(std::filesystem::temp_directory_path() / "yolov5n_demo_draw_input.ppm");
    const auto output_path = std::filesystem::temp_directory_path() / "yolov5n_demo_draw_output.ppm";

    feather::demo::ImageData image;
    ASSERT_EQ(feather::demo::LoadImage(input_path.string(), &image), 0);

    std::vector<feather::demo::Detection> detections = {
        feather::demo::Detection{0, 0.95f, 0.0f, 0.0f, 1.0f, 1.0f},
    };

    ASSERT_EQ(feather::demo::SaveDetectionsImage(image, detections, output_path.string()), 0);
    ASSERT_TRUE(std::filesystem::exists(output_path));
    ASSERT_GT(std::filesystem::file_size(output_path), 0U);

    feather::demo::ImageData annotated;
    ASSERT_EQ(feather::demo::LoadImage(output_path.string(), &annotated), 0);
    ASSERT_EQ(annotated.width, image.width);
    ASSERT_EQ(annotated.height, image.height);
    EXPECT_NE(annotated.pixels, image.pixels);
}
