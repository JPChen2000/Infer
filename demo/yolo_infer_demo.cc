// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "demo/yolo_runner.h"
#include "util/fp16.h"

namespace {
void Usage() {
    std::cerr << "usage: yolo_demo --model <model.fth> --image <image> [--output output.jpg]"
                 " [--backend host|common|x86|cuda] [--layout auto|nchw|nhwc]"
                 " [--conf-thresh 0.25] [--iou-thresh 0.45] [--max-detections 300]"
                 " [--json <raw-and-detections.json>] [--dump-input <fp32-nchw.bin>] [--profile]\n";
}

bool Threshold(const std::string& text, float* value) {
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (text.empty() || end != text.c_str() + text.size() || errno == ERANGE ||
        !std::isfinite(parsed) || parsed < 0 || parsed > 1) return false;
    *value = parsed;
    return true;
}

bool PositiveSize(const std::string& text, size_t* value) {
    size_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0) return false;
    *value = parsed;
    return true;
}

bool SaveInput(const std::string& path, const feather::Tensor* input) {
    if (!input || (input->data_type() != feather::DataType::FP32 && input->data_type() != feather::DataType::FP16)) return false;
    feather::ImageShape4D shape;
    if (!feather::DecodeImageShape4D(input->dims().data(), input->layout(), &shape)) return false;
    std::vector<float> nchw;
    nchw.reserve(static_cast<size_t>(input->numel()));
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < shape.c; ++c) {
            for (int64_t y = 0; y < shape.h; ++y) {
                for (int64_t x = 0; x < shape.w; ++x) {
                    const int64_t index = feather::OffsetForImage4D(input->layout(), n, c, y, x, shape.c, shape.h, shape.w);
                    nchw.push_back(input->data_type() == feather::DataType::FP16
                        ? feather::HalfToFloat(input->data<uint16_t>()[index]) : input->data<float>()[index]);
                }
            }
        }
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(nchw.data()), static_cast<std::streamsize>(nchw.size() * sizeof(float)));
    out.close();
    return out.good();
}

bool SaveJson(const std::string& path, const feather::demo::YoloRunner& runner,
               const feather::demo::ImageData& image, const std::vector<feather::demo::Detection>& detections) {
    std::ofstream out(path);
    if (!out) return false;
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    const auto& metadata = runner.DetectionMetadata();
    const auto& letterbox = runner.LastLetterbox();
    feather::ImageShape4D input_shape;
    const auto* input = runner.LastInput();
    if (!input || !feather::DecodeImageShape4D(input->dims().data(), input->layout(), &input_shape)) return false;
    out << "{\"family\":\"" << metadata.family << "\",\"output_format\":\"" << metadata.output_format
        << "\",\"output_layout\":\"" << metadata.output_layout
        << "\",\"num_classes\":" << metadata.num_classes
        << ",\"nms_required\":" << (metadata.nms_required ? "true" : "false")
        << ",\"input_shape\":[" << input_shape.n << ',' << input_shape.c << ',' << input_shape.h << ',' << input_shape.w << ']'
        << ",\"image_size\":[" << image.width << ',' << image.height << ']'
        << ",\"letterbox\":{\"scale\":" << letterbox.scale << ",\"pad_x\":" << letterbox.pad_x << ",\"pad_y\":" << letterbox.pad_y << '}'
        << ",\"raw_output\":{\"shape\":[";
    const auto* raw = runner.LastRawOutput();
    if (!raw) return false;
    const auto& shape = raw->dims().data();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << "],\"data\":[";
    for (int64_t i = 0; i < raw->numel(); ++i) {
        if (i) out << ',';
        const float value = raw->data_type() == feather::DataType::FP16
                                ? feather::HalfToFloat(raw->data<uint16_t>()[i]) : raw->data<float>()[i];
        if (std::isfinite(value)) out << value;
        else out << "null";
    }
    out << "]},\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        if (i) out << ',';
        const auto& d = detections[i];
        out << "{\"class_id\":" << d.class_id << ",\"score\":" << d.score
            << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << "]}";
    }
    out << "]}\n";
    out.close();
    return out.good();
}

void Profile(const feather::demo::YoloRunner& runner) {
    auto summaries = runner.RuntimeProfileSummaries();
    std::stable_sort(summaries.begin(), summaries.end(), [](const auto& a, const auto& b) { return a.total_ms > b.total_ms; });
    for (size_t i = 0; i < std::min<size_t>(summaries.size(), 30); ++i) {
        const auto& item = summaries[i];
        std::cout << "[profile] node=" << item.node_name << " op=" << item.op_type
                  << " calls=" << item.call_count << " total_ms=" << item.total_ms << " avg_ms=" << item.avg_ms << '\n';
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string model_path, image_path, json_path, input_dump_path;
    std::string output_path = "output.jpg";
    feather::demo::YoloBackend backend = feather::demo::YoloBackend::kHost;
    feather::demo::YoloLayoutOverride layout = feather::demo::YoloLayoutOverride::kAuto;
    float conf_thresh = 0.25f, iou_thresh = 0.45f;
    size_t max_detections = 300;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") { Usage(); return 0; }
        if (argument == "--profile") { profile = true; continue; }
        if (i + 1 >= argc) { std::cerr << "missing value for " << argument << '\n'; Usage(); return 1; }
        const std::string value = argv[++i];
        bool valid = true;
        if (argument == "--model") model_path = value;
        else if (argument == "--image") image_path = value;
        else if (argument == "--output") output_path = value;
        else if (argument == "--json") json_path = value;
        else if (argument == "--dump-input") input_dump_path = value;
        else if (argument == "--backend") valid = feather::demo::ParseYoloBackend(value, &backend);
        else if (argument == "--layout") valid = feather::demo::ParseYoloLayoutOverride(value, &layout);
        else if (argument == "--conf-thresh") valid = Threshold(value, &conf_thresh);
        else if (argument == "--iou-thresh") valid = Threshold(value, &iou_thresh);
        else if (argument == "--max-detections") valid = PositiveSize(value, &max_detections);
        else valid = false;
        if (!valid) { std::cerr << "invalid argument or value: " << argument << ' ' << value << '\n'; Usage(); return 1; }
    }
    if (model_path.empty() || image_path.empty() || output_path.empty()) { Usage(); return 1; }
    feather::demo::YoloRunner runner;
    if (runner.Load(model_path, backend, layout) != 0) {
        std::cerr << "failed to load YOLO model: " << runner.LastError() << '\n';
        return 1;
    }
    runner.SetMaxDetections(max_detections);
    runner.SetRuntimeProfilingEnabled(profile);
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';
    std::vector<feather::demo::Detection> detections;
    if (runner.Run(image_path, conf_thresh, iou_thresh, &detections) != 0) {
        std::cerr << "YOLO inference failed: " << runner.LastError() << '\n';
        return 1;
    }
    std::cout << "[framework] " << runner.DescribeLastRun() << '\n';
    std::cout << "family=" << runner.DetectionMetadata().family << " detections=" << detections.size() << '\n';
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& detection : detections) {
        std::cout << "class=" << detection.class_id << " score=" << detection.score << " box=["
                  << detection.x1 << ',' << detection.y1 << ',' << detection.x2 << ',' << detection.y2 << "]\n";
    }
    if (profile) Profile(runner);
    if (!input_dump_path.empty() && !SaveInput(input_dump_path, runner.LastInput())) {
        std::cerr << "failed to dump YOLO input: " << input_dump_path << '\n';
        return 1;
    }
    feather::demo::ImageData image;
    if (feather::demo::LoadImage(image_path, &image) != 0) {
        std::cerr << "failed to load YOLO image for annotation: " << image_path << '\n';
        return 1;
    }
    if (!json_path.empty() && !SaveJson(json_path, runner, image, detections)) {
        std::cerr << "failed to write YOLO JSON output: " << json_path << '\n';
        return 1;
    }
    if (feather::demo::SaveDetectionsImage(image, detections, output_path) != 0) {
        std::cerr << "failed to save annotated YOLO image: " << output_path << '\n';
        return 1;
    }
    std::cout << "saved annotated image: " << output_path << '\n';
    return 0;
}
