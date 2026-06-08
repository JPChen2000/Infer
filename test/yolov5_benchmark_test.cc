#include <gtest/gtest.h>

#include "demo/yolov5_benchmark_cli.h"
#include "demo/yolov5_runner.h"

TEST(yolov5_benchmark_test, ParseCommandLineKeepsBenchmarkArgsAndRequiredPaths) {
    char arg0[] = "yolov5_benchmark_demo";
    char arg1[] = "--model";
    char arg2[] = "model.fth";
    char arg3[] = "--image";
    char arg4[] = "image.ppm";
    char arg5[] = "--benchmarks=5";
    char arg6[] = "--benchmark_repetitions=10";
    char arg7[] = "--conf-thresh";
    char arg8[] = "0.33";
    char arg9[] = "--iou-thresh";
    char arg10[] = "0.44";
    char* args[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10};
    char** argv = args;
    int argc = 11;

    feather::demo::Yolov5BenchmarkCommandLine options;
    ASSERT_TRUE(feather::demo::ParseYolov5BenchmarkCommandLine(&argc, &argv, &options));
    EXPECT_EQ(options.model_path, "model.fth");
    EXPECT_EQ(options.image_path, "image.ppm");
    EXPECT_FLOAT_EQ(options.conf_thresh, 0.33f);
    EXPECT_FLOAT_EQ(options.iou_thresh, 0.44f);
    EXPECT_EQ(argc, 3);
    EXPECT_STREQ(argv[1], "--benchmarks=5");
    EXPECT_STREQ(argv[2], "--benchmark_repetitions=10");
}

TEST(yolov5_benchmark_test, RunPreparedImageRejectsUnloadedRunner) {
    feather::demo::Yolov5Runner runner;
    feather::demo::ImageData image;
    std::vector<feather::demo::Detection> detections;

    EXPECT_EQ(runner.RunPreparedImage(image, 0.25f, 0.45f, &detections), -1);
}
