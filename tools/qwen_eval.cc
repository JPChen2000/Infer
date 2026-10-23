// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
// Offline evaluation; score/calibration timing is never used as decode throughput.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "demo/qwen_runner.h"
#include "model/model_io.h"
#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using feather::demo::QwenRunner;
struct Range {
    uint64_t count{};
    double amax{}, square_sum{};
    void Add(const std::vector<float>& values) {
        for (const double value : values) {
            if (!std::isfinite(value)) throw std::runtime_error("non-finite calibration activation");
            amax = std::max(amax, std::abs(value));
            square_sum += value * value;
        }
        count += values.size();
    }
};

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<std::vector<int64_t>> Documents(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read token input");
    std::vector<std::vector<int64_t>> result;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::vector<int64_t> ids;
        int64_t token;
        while (stream >> token) {
            if (token < 0) throw std::runtime_error("negative token ID");
            ids.push_back(token);
        }
        if (!stream.eof() || ids.size() < 2) throw std::runtime_error("each line requires at least two integer IDs");
        result.push_back(std::move(ids));
    }
    if (result.empty()) throw std::runtime_error("empty evaluation corpus");
    return result;
}

std::string Metadata(const feather::model::ModelDesc& model, const std::string& key) {
    const auto found = model.metadata.find(key);
    const auto* value = found == model.metadata.end() ? nullptr : std::get_if<std::string>(&found->second);
    if (!value) throw std::runtime_error("model requires metadata " + key + "; re-export the BF16 model");
    return *value;
}

void Check(int status, const QwenRunner& runner) {
    if (status != 0) throw std::runtime_error(runner.LastError());
}

size_t DeviceUsed() {
#ifdef FEATHER_WITH_CUDA
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) return total - free;
#endif
    return 0;
}

int Run(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "qwen_eval --model FILE --input TOKEN_LINES --output DIRECTORY "
                         "[--mode score|calibrate|benchmark] [--backend cuda|common|x86] "
                         "[--context N] [--new-tokens N] [--repeats N] [--fp8-weight-cache-mb N]\n";
            return 0;
        }
        if (i + 1 == argc || args.count(key) ||
            (key != "--model" && key != "--input" && key != "--output" && key != "--mode" &&
             key != "--backend" && key != "--context" && key != "--new-tokens" && key != "--repeats" && key != "--fp8-weight-cache-mb"))
            throw std::runtime_error("unknown, duplicate or missing CLI argument: " + key);
        args[key] = argv[i + 1];
    }
    for (const auto& required : {"--model", "--input", "--output"})
        if (!args.count(required)) throw std::runtime_error(std::string("missing ") + required);
    const auto number = [&](const std::string& key, int64_t fallback) {
        if (!args.count(key)) return fallback;
        size_t end = 0;
        const auto value = std::stoll(args[key], &end);
        if (end != args[key].size() || value < 0) throw std::runtime_error("invalid integer: " + key);
        return static_cast<int64_t>(value);
    };
    const std::string mode = args.count("--mode") ? args["--mode"] : "score";
    if (mode != "score" && mode != "calibrate" && mode != "benchmark") throw std::runtime_error("unknown evaluation mode");
    const int64_t generated_count = number("--new-tokens", 32), repeats = number("--repeats", 3);
    if (generated_count < 2 || generated_count > 100000 || repeats < 1 || repeats > 100)
        throw std::runtime_error("new-tokens must be in [2,100000], repeats in [1,100]");
    auto backend = feather::demo::QwenBackend::kCuda;
    if (args.count("--backend") && !feather::demo::ParseQwenBackend(args["--backend"], &backend))
        throw std::runtime_error("unknown backend");
    auto documents = Documents(args["--input"]);
    feather::model::ModelLoader loader;
    if (!loader.Load(args["--model"])) throw std::runtime_error("cannot load model descriptor");
    int64_t vocabulary = 0;
    for (const auto& value : loader.model().graph.values)
        if (value.tensor.name == "logits" && !value.tensor.dims.empty()) vocabulary = value.tensor.dims.back();
    if (vocabulary <= 1) throw std::runtime_error("evaluation requires vocabulary logits");
    for (const auto& doc : documents) for (auto id : doc)
        if (id >= vocabulary) throw std::runtime_error("token ID exceeds model vocabulary");
    feather::demo::QwenLoadOptions options;
    options.context_size = number("--context", 0);
    const auto cache_mb = number("--fp8-weight-cache-mb", 256);
    if (static_cast<uint64_t>(cache_mb) > std::numeric_limits<size_t>::max() / (1024 * 1024))
        throw std::runtime_error("FP8 weight cache size overflow");
    options.fp8_weight_cache_bytes = static_cast<size_t>(cache_mb) * 1024 * 1024;
    options.retain_logits = mode != "benchmark";
    std::map<std::string, std::pair<std::string, std::string>> projections;
    std::map<std::string, Range> ranges;
    std::string checkpoint_hash, config_hash;
    if (mode == "calibrate") {
        checkpoint_hash = Metadata(loader.model(), "qwen.source.checkpoint_sha256");
        config_hash = Metadata(loader.model(), "qwen.source.config_sha256");
        for (const auto& hash : {checkpoint_hash, config_hash})
            if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::runtime_error("invalid source model fingerprint");
        for (const auto& value : loader.model().graph.values)
            if (value.tensor.data_type == feather::DataType::FP8E4M3 || value.tensor.data_type == feather::DataType::FP8E5M2)
                throw std::runtime_error("calibration requires an unquantized BF16 source graph");
        const std::string prefix = "qwen.projection.input.";
        for (const auto& entry : loader.model().metadata) if (entry.first.compare(0, prefix.size(), prefix) == 0) {
            const auto key = entry.first.substr(prefix.size());
            auto input = Metadata(loader.model(), entry.first);
            auto output = Metadata(loader.model(), "qwen.projection.output." + key);
            projections[key] = {input, output};
            ranges.emplace(input, Range{}); ranges.emplace(output, Range{});
        }
        if (projections.empty()) throw std::runtime_error("projection metadata absent; re-export the BF16 graph");
        for (const auto& range : ranges) options.additional_outputs.push_back(range.first);
    }
    const std::filesystem::path output(args["--output"]);
    // The Python wrapper places manifests here before invoking this binary.
    // Reject runtime artifacts so a direct rerun cannot append old samples.
    for (const auto& artifact : {"runtime.json", "scores.tsv", "benchmark.tsv", "ranges.tsv", "samples"})
        if (std::filesystem::exists(output / artifact))
            throw std::runtime_error("evaluation output already contains runtime artifacts");
    const auto initial_device = backend == feather::demo::QwenBackend::kCuda ? DeviceUsed() : 0;
    QwenRunner runner;
    auto start = Clock::now();
    Check(runner.Load(args["--model"], backend, options), runner);
    const double load_ms = Milliseconds(start);
    for (const auto& doc : documents) {
        const uint64_t needed = doc.size() + (mode == "benchmark" ? generated_count : 0);
        if (needed > static_cast<uint64_t>(options.context_size ? runner.MaxContext() : runner.ModelContextLimit()))
            throw std::runtime_error("document exceeds evaluation capacity; change --context or dataset");
    }
    std::filesystem::create_directories(output);
    std::ofstream scores(output / "scores.tsv"), benchmarks(output / "benchmark.tsv");
    scores << "document\tposition\ttarget\tgreedy\tnll\n" << std::setprecision(17);
    benchmarks << "document\trun\tttft_ms\tprefill_ms\tdecode_ms\tdecode_steps\ttotal_ms\tcuda_graph_launches\tdevice_used_bytes\n"
               << std::setprecision(17);
    if (!scores || !benchmarks) throw std::runtime_error("cannot open evaluation outputs");
    uint64_t count = 0, correct = 0;
    double total_nll = 0;
    size_t observed_device = initial_device;
    for (size_t d = 0; d < documents.size(); ++d) {
        const auto& ids = documents[d];
        if (mode == "benchmark") {
            std::vector<int64_t> reference_tokens;
            for (int64_t run = 0; run <= repeats; ++run) {
                Check(runner.Reset(), runner);
                std::vector<double> callbacks;
                std::vector<int64_t> generated;
                double prefill = 0;
                const auto begin = Clock::now();
                Check(runner.GenerateStream(ids, static_cast<int>(generated_count), {}, [&](int64_t id) {
                    callbacks.push_back(Milliseconds(begin)); generated.push_back(id);
                    if (callbacks.size() == 1) prefill = runner.LastPrefillMilliseconds();
                }), runner);
                const double total = Milliseconds(begin);
                if (generated.size() != static_cast<size_t>(generated_count)) throw std::runtime_error("benchmark generation truncated");
                if (run == 0) reference_tokens = generated;
                else if (generated != reference_tokens) throw std::runtime_error("generation changed between repeated runs");
                const size_t used = backend == feather::demo::QwenBackend::kCuda ? DeviceUsed() : 0;
                observed_device = std::max(observed_device, used);
                benchmarks << d << '\t' << run << '\t' << callbacks.front() << '\t' << prefill << '\t'
                           << callbacks.back() - callbacks.front() << '\t' << callbacks.size() - 1 << '\t'
                           << total << '\t' << runner.CudaGraphLaunchCount() << '\t' << used << '\n';
                if (run == 0) {
                    std::ofstream tokens(output / ("generated_" + std::to_string(d) + ".txt"));
                    for (auto id : generated) tokens << id << ' ';
                }
            }
        } else {
            Check(runner.Reset(), runner);
            std::vector<float> logits, values;
            std::map<std::string, std::vector<float>> final_activations;
            const size_t steps = ids.size() - (mode == "score" ? 1 : 0);
            for (size_t i = 0; i < steps; ++i) {
                Check(runner.EvaluateToken(ids[i], &logits), runner);
                if (logits.size() != static_cast<size_t>(vocabulary)) throw std::runtime_error("logits vocabulary changed");
                if (mode == "score") {
                    const auto best = std::max_element(logits.begin(), logits.end());
                    const double maximum = *best;
                    double sum = 0;
                    for (const auto value : logits) sum += std::exp(static_cast<double>(value) - maximum);
                    const double nll = maximum + std::log(sum) - logits[static_cast<size_t>(ids[i + 1])];
                    const auto greedy = std::distance(logits.begin(), best);
                    total_nll += nll; ++count; correct += greedy == ids[i + 1];
                    scores << d << '\t' << i + 1 << '\t' << ids[i + 1] << '\t' << greedy << '\t' << nll << '\n';
                } else {
                    for (auto& range : ranges) {
                        Check(runner.ReadOutput(range.first, &values), runner);
                        range.second.Add(values);
                        if (i + 1 == steps) final_activations[range.first] = values;
                    }
                    ++count;
                }
            }
            if (mode == "score") {
                std::ofstream raw(output / ("final_" + std::to_string(d) + ".fp32"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                if (!raw) throw std::runtime_error("failed writing final logits");
            } else {
                std::filesystem::create_directories(output / "samples");
                for (const auto& projection : projections) {
                    if (projection.first.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._") != std::string::npos)
                        throw std::runtime_error("unsafe projection sample name");
                    for (const auto& side : {std::make_pair("input", projection.second.first), std::make_pair("output", projection.second.second)}) {
                        const auto& sample = final_activations.at(side.second);
                        std::ofstream raw(output / "samples" / (projection.first + "." + side.first + ".fp32"),
                                          std::ios::binary | std::ios::app);
                        raw.write(reinterpret_cast<const char*>(sample.data()), sample.size() * sizeof(float));
                        if (!raw) throw std::runtime_error("failed writing calibration sensitivity sample");
                    }
                }
            }
        }
        if (backend == feather::demo::QwenBackend::kCuda) observed_device = std::max(observed_device, DeviceUsed());
        std::cerr << "completed document " << d + 1 << '/' << documents.size() << '\n';
    }
    if (mode == "calibrate") {
        std::ofstream table(output / "ranges.tsv");
        table << "projection\tside\tamax\trms\tcount\n" << std::setprecision(17);
        for (const auto& projection : projections) {
            for (const auto& side : {std::make_pair("input", projection.second.first), std::make_pair("output", projection.second.second)}) {
                const auto& range = ranges.at(side.second);
                if (!range.count) throw std::runtime_error("calibration range not observed");
                table << projection.first << '\t' << side.first << '\t' << range.amax << '\t'
                      << std::sqrt(range.square_sum / range.count) << '\t' << range.count << '\n';
            }
        }
        if (!table) throw std::runtime_error("failed writing ranges");
    }
    std::ofstream summary(output / "runtime.json");
    summary << std::setprecision(17) << "{\n  \"version\": 1,\n  \"mode\": \"" << mode
            << "\",\n  \"documents\": " << documents.size() << ",\n  \"tokens\": " << count
            << ",\n  \"total_nll\": " << total_nll << ",\n  \"target_top1_correct\": " << correct
            << ",\n  \"load_ms\": " << load_ms << ",\n  \"final_context_capacity\": " << runner.MaxContext()
            << ",\n  \"context_growths\": " << runner.ContextGrowthCount()
            << ",\n  \"device_initial_used_bytes\": " << initial_device
            << ",\n  \"device_max_observed_used_bytes\": " << observed_device;
    if (mode == "calibrate") summary << ",\n  \"checkpoint_sha256\": \"" << checkpoint_hash
        << "\",\n  \"model_config_sha256\": \"" << config_hash << '"';
    summary << "\n}\n";
    if (!summary || !scores || !benchmarks) throw std::runtime_error("failed writing evaluation outputs");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return Run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "qwen_eval: " << error.what() << '\n'; return 1; }
}
