#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "demo/image_io.h"
#include "model/model_io.h"
#include "util/bf16.h"

namespace {

struct Range {
    static constexpr size_t kReservoirCapacity = 2048;
    float minimum{std::numeric_limits<float>::infinity()};
    float maximum{std::numeric_limits<float>::lowest()};
    bool observed{false};
    uint64_t sample_count{0};
    uint64_t sample_state{0x9e3779b97f4a7c15ULL};
    std::vector<float> samples;
};

void Usage(const char* program) {
    std::cerr << "usage: " << program
              << " --input MODEL --output TABLE --image IMAGE [--image IMAGE ...] [--asymmetric]"
              << " [--percentile COVERAGE] [--mse]\n";
}

bool ValueAfter(int argc, char** argv, int* index, std::string* value) {
    if (index == nullptr || value == nullptr || *index + 1 >= argc) return false;
    *value = argv[++*index];
    return true;
}

float DecodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t result = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            const float value = std::ldexp(static_cast<float>(fraction), -24);
            return sign == 0 ? value : -value;
        }
    } else if (exponent == 0x1fu) {
        result |= 0x7f800000u | (fraction << 13);
    } else {
        result |= ((exponent + 112u) << 23) | (fraction << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

float ReadFloat(const feather::Tensor& tensor, size_t index) {
    switch (tensor.data_type()) {
        case feather::DataType::FP32:
            return tensor.data<float>()[index];
        case feather::DataType::FP16:
            return DecodeFloat16(tensor.data<uint16_t>()[index]);
        case feather::DataType::BF16: {
            const uint32_t bits = static_cast<uint32_t>(tensor.data<feather::BFloat16>()[index].bits) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

bool IsFloatType(feather::DataType type) {
    return type == feather::DataType::FP32 || type == feather::DataType::FP16 ||
           type == feather::DataType::BF16;
}

void Observe(const feather::Tensor& tensor, Range* range) {
    if (range == nullptr || !tensor.IsInitialized() || !IsFloatType(tensor.data_type()) || tensor.numel() <= 0) {
        return;
    }
    for (int64_t index = 0; index < tensor.numel(); ++index) {
        const float value = ReadFloat(tensor, static_cast<size_t>(index));
        if (!std::isfinite(value)) continue;
        range->minimum = std::min(range->minimum, value);
        range->maximum = std::max(range->maximum, value);
        range->observed = true;
        ++range->sample_count;
        if (range->samples.size() < Range::kReservoirCapacity) {
            range->samples.push_back(value);
        } else {
            // Deterministic reservoir sampling keeps calibration bounded while
            // retaining values from every tensor and every calibration image.
            range->sample_state = range->sample_state * 6364136223846793005ULL + 1442695040888963407ULL;
            const uint64_t slot = range->sample_state % range->sample_count;
            if (slot < Range::kReservoirCapacity) range->samples[static_cast<size_t>(slot)] = value;
        }
    }
}

const feather::model::ValueDesc* FindValue(const feather::model::ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) return &value;
    }
    return nullptr;
}

bool ComputeAffineParams(float minimum, float maximum, bool asymmetric, float* scale, int32_t* zero_point) {
    if (scale == nullptr || zero_point == nullptr || !std::isfinite(minimum) || !std::isfinite(maximum) ||
        maximum < minimum) {
        return false;
    }
    *scale = 1.0f;
    *zero_point = 0;
    if (asymmetric) {
        if (maximum != minimum) {
            *scale = (maximum - minimum) / 255.0f;
            *zero_point = static_cast<int32_t>(std::round(-128.0f - minimum / *scale));
            *zero_point = std::max(-128, std::min(127, *zero_point));
        }
    } else {
        const float maximum_abs = std::max(std::fabs(minimum), std::fabs(maximum));
        *scale = maximum_abs == 0.0f ? 1.0f : maximum_abs / 127.0f;
    }
    return std::isfinite(*scale) && *scale > 0.0f;
}

bool OptimizeRangeMse(const Range& range, bool asymmetric, float* minimum, float* maximum) {
    if (minimum == nullptr || maximum == nullptr || range.samples.size() < 32) return false;
    std::vector<float> samples = range.samples;
    std::sort(samples.begin(), samples.end());
    // Include the full observed range and search only modest tail clipping.
    // The full-range candidate prevents the optional method from silently
    // discarding rare but important activations when it does not reduce MSE.
    constexpr std::array<double, 8> kTails = {0.0, 0.0001, 0.0005, 0.001,
                                              0.002, 0.005, 0.01, 0.02};
    double best_error = std::numeric_limits<double>::infinity();
    float best_minimum = range.minimum;
    float best_maximum = range.maximum;

    auto evaluate = [&](float candidate_minimum, float candidate_maximum) {
        float scale = 1.0f;
        int32_t zero_point = 0;
        if (!(candidate_maximum > candidate_minimum) ||
            !ComputeAffineParams(candidate_minimum, candidate_maximum, asymmetric, &scale, &zero_point)) {
            return;
        }
        double error = 0.0;
        for (const float value : samples) {
            const double transformed = static_cast<double>(value) / scale + zero_point;
            const double rounded = std::round(transformed);
            const double quantized = std::max(-128.0, std::min(127.0, rounded));
            const double reconstructed = (quantized - zero_point) * scale;
            const double difference = static_cast<double>(value) - reconstructed;
            error += difference * difference;
        }
        if (error < best_error) {
            best_error = error;
            best_minimum = candidate_minimum;
            best_maximum = candidate_maximum;
        }
    };

    evaluate(range.minimum, range.maximum);
    if (asymmetric) {
        for (const double lower_tail : kTails) {
            for (const double upper_tail : kTails) {
                const size_t lower_index = static_cast<size_t>(
                    std::floor(lower_tail * static_cast<double>(samples.size() - 1)));
                const size_t upper_index = static_cast<size_t>(
                    std::ceil((1.0 - upper_tail) * static_cast<double>(samples.size() - 1)));
                if (lower_index < upper_index) {
                    evaluate(samples[lower_index], samples[std::min(upper_index, samples.size() - 1)]);
                }
            }
        }
    } else {
        std::vector<float> absolute_samples;
        absolute_samples.reserve(samples.size());
        for (const float value : samples) absolute_samples.push_back(std::fabs(value));
        std::sort(absolute_samples.begin(), absolute_samples.end());
        for (const double tail : kTails) {
            const size_t index = static_cast<size_t>(
                std::floor((1.0 - tail) * static_cast<double>(absolute_samples.size() - 1)));
            const float maximum_abs = absolute_samples[std::min(index, absolute_samples.size() - 1)];
            evaluate(-maximum_abs, maximum_abs);
        }
    }
    if (!std::isfinite(best_error) || !(best_maximum > best_minimum)) return false;
    *minimum = best_minimum;
    *maximum = best_maximum;
    return true;
}

bool WriteTable(const std::string& path, const feather::model::ModelDesc& model,
                const std::unordered_map<std::string, Range>& ranges, bool asymmetric,
                bool use_mse, float percentile, size_t* written, size_t* missing) {
    if (written == nullptr || missing == nullptr) return false;
    *written = 0;
    *missing = 0;
    std::ofstream output(path);
    if (!output.good()) return false;
    output << "# value_name scale zero_point\n";
    for (const auto& value : model.graph.values) {
        const auto it = ranges.find(value.tensor.name);
        if (it == ranges.end() || !it->second.observed) {
            ++*missing;
            continue;
        }
        float minimum = it->second.minimum;
        float maximum = it->second.maximum;
        if (use_mse) {
            float optimized_minimum = minimum;
            float optimized_maximum = maximum;
            if (OptimizeRangeMse(it->second, asymmetric, &optimized_minimum, &optimized_maximum)) {
                minimum = optimized_minimum;
                maximum = optimized_maximum;
            }
        } else if (percentile < 1.0f && it->second.samples.size() >= 2) {
            std::vector<float> samples = it->second.samples;
            std::sort(samples.begin(), samples.end());
            if (asymmetric) {
                const double tail = (1.0 - static_cast<double>(percentile)) * 0.5;
                const size_t lower_index = static_cast<size_t>(std::floor(tail * static_cast<double>(samples.size() - 1)));
                const size_t upper_index = static_cast<size_t>(std::ceil((1.0 - tail) * static_cast<double>(samples.size() - 1)));
                minimum = samples[std::min(lower_index, samples.size() - 1)];
                maximum = samples[std::min(upper_index, samples.size() - 1)];
            } else {
                std::vector<float> absolute_samples;
                absolute_samples.reserve(samples.size());
                for (const float sample : samples) absolute_samples.push_back(std::fabs(sample));
                std::sort(absolute_samples.begin(), absolute_samples.end());
                const size_t index = static_cast<size_t>(std::ceil(static_cast<double>(percentile) *
                                                                    static_cast<double>(absolute_samples.size() - 1)));
                const float maximum_abs = absolute_samples[std::min(index, absolute_samples.size() - 1)];
                minimum = -maximum_abs;
                maximum = maximum_abs;
            }
            if (!(maximum > minimum)) {
                minimum = it->second.minimum;
                maximum = it->second.maximum;
            }
        }
        float scale = 1.0f;
        int32_t zero_point = 0;
        if (!ComputeAffineParams(minimum, maximum, asymmetric, &scale, &zero_point)) return false;
        output << value.tensor.name << ' ' << scale << ' ' << zero_point << '\n';
        ++*written;
    }
    return output.good();
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::vector<std::string> image_paths;
    bool asymmetric = false;
    bool use_mse = false;
    float percentile = 1.0f;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input") {
            if (!ValueAfter(argc, argv, &index, &input_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--output") {
            if (!ValueAfter(argc, argv, &index, &output_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--image") {
            std::string image_path;
            if (!ValueAfter(argc, argv, &index, &image_path)) { Usage(argv[0]); return 2; }
            image_paths.push_back(std::move(image_path));
        } else if (argument == "--asymmetric") {
            asymmetric = true;
        } else if (argument == "--mse") {
            use_mse = true;
        } else if (argument == "--percentile") {
            std::string value;
            if (!ValueAfter(argc, argv, &index, &value)) { Usage(argv[0]); return 2; }
            try {
                percentile = std::stof(value);
            } catch (...) {
                Usage(argv[0]);
                return 2;
            }
        } else {
            Usage(argv[0]);
            return 2;
        }
    }
    if (input_path.empty() || output_path.empty() || image_paths.empty() ||
        !std::isfinite(percentile) || percentile <= 0.0f || percentile > 1.0f) {
        Usage(argv[0]);
        return 2;
    }

    feather::model::ModelLoader loader;
    if (!loader.Load(input_path)) {
        std::cerr << "failed to load model: " << input_path << '\n';
        return 3;
    }
    const auto& model = loader.model();
    if (model.graph.inputs.empty()) {
        std::cerr << "model has no graph input\n";
        return 4;
    }
    const std::string input_name = model.graph.inputs.front();
    const auto* input_value = FindValue(model, input_name);
    if (input_value == nullptr ||
        (input_value->tensor.data_type != feather::DataType::FP32 &&
         input_value->tensor.data_type != feather::DataType::FP16)) {
        std::cerr << "calibration requires an FP32 or FP16 image input\n";
        return 4;
    }
    if (input_value->tensor.dims.size() != 4) {
        std::cerr << "calibration currently requires a rank-4 image input\n";
        return 4;
    }
    feather::ImageShape4D input_shape;
    if (!feather::DecodeImageShape4D(input_value->tensor.dims, input_value->tensor.layout, &input_shape) ||
        input_shape.n != 1 || input_shape.c != 3 || input_shape.h <= 0 || input_shape.h != input_shape.w) {
        std::cerr << "calibration currently requires an N=1, C=3 square image input\n";
        return 4;
    }

    feather::StaticGraph static_graph;
    static_graph.SetKernelDevice(feather::DeviceType::COMMON);
    if (static_graph.SetModel(model) != 0) return 5;
    static_graph.SetPassManager(nullptr);
    for (const auto& value : model.graph.values) {
        if (!value.constant) continue;
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph.SetTensor(value.tensor.name, std::move(tensor)) != 0) {
            std::cerr << "failed to bind constant: " << value.tensor.name << '\n';
            return 5;
        }
    }
    auto input_tensor = std::make_shared<feather::Tensor>(input_value->tensor.dims);
    input_tensor->set_layout(input_value->tensor.layout);
    input_tensor->set_data_type(input_value->tensor.data_type);
    if (input_value->tensor.data_type == feather::DataType::FP16) {
        (void)input_tensor->mutable_data<uint16_t>();
    } else {
        (void)input_tensor->mutable_data<float>();
    }
    if (static_graph.SetTensor(input_name, input_tensor) != 0 || static_graph.Build() != 0) {
        std::cerr << "failed to build calibration graph\n";
        return 5;
    }
    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    if (lowering.Lower(static_graph, &runtime_graph) != 0) {
        std::cerr << "failed to lower calibration graph\n";
        return 5;
    }

    std::unordered_map<std::string, Range> ranges;
    for (const auto& image_path : image_paths) {
        feather::demo::ImageData image;
        if (feather::demo::LoadImage(image_path, &image) != 0) {
            std::cerr << "failed to load image: " << image_path << '\n';
            return 6;
        }
        feather::demo::LetterboxInfo letterbox;
        if (feather::demo::PreprocessImageToTensor(image, static_cast<int>(input_shape.h),
                                                   input_value->tensor.data_type, input_tensor.get(), &letterbox) != 0) {
            std::cerr << "failed to preprocess image: " << image_path << '\n';
            return 6;
        }
        if (runtime_graph.Run() != 0) {
            std::cerr << "calibration graph execution failed for image: " << image_path << '\n';
            return 7;
        }
        for (const auto& value : model.graph.values) {
            auto tensor = runtime_graph.GetTensor(value.tensor.name);
            if (tensor != nullptr) Observe(*tensor, &ranges[value.tensor.name]);
        }
    }

    size_t written = 0;
    size_t missing = 0;
    if (!WriteTable(output_path, model, ranges, asymmetric, use_mse, percentile, &written, &missing) || written == 0) {
        std::cerr << "failed to write calibration table: " << output_path << '\n';
        return 8;
    }
    std::cout << "calibrated_values=" << written << " missing_values=" << missing
              << " images=" << image_paths.size() << " calibration="
              << (use_mse ? "mse" : (percentile < 1.0f ? "percentile" : "minmax"))
              << " percentile=" << percentile
              << " table=" << output_path << '\n';
    return 0;
}
