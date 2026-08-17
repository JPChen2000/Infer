#include "demo/classification_demo.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "demo/classification_runner.h"

namespace feather {
namespace demo {

namespace {

void PrintUsage(const char* demo_name) {
    std::cerr << "usage: " << demo_name
              << " --model <model.fth> --image <image> [--labels <labels.txt>]"
                 " [--top-k 5] [--backend host|common|x86|cuda] [--profile]\n";
}

void PrintProfile(const ClassificationRunner& runner) {
    auto summaries = runner.RuntimeProfileSummaries();
    std::sort(summaries.begin(), summaries.end(),
              [](const RuntimeProfileSummary& lhs, const RuntimeProfileSummary& rhs) {
                  return lhs.total_ms > rhs.total_ms;
              });
    std::cout << "[profile] top_runtime_nodes=" << summaries.size() << '\n';
    for (size_t i = 0; i < summaries.size(); ++i) {
        const auto& item = summaries[i];
        std::cout << "[profile] #" << (i + 1) << " node=" << item.node_name
                  << " op=" << item.op_type << " calls=" << item.call_count
                  << " total_ms=" << item.total_ms << " avg_ms=" << item.avg_ms << '\n';
    }
}

}  // namespace

int RunClassificationDemo(int argc, char** argv, const char* demo_name) {
    std::string model_path;
    std::string image_path;
    std::string labels_path;
    ClassificationBackend backend = ClassificationBackend::kHost;
    int top_k = 5;
    bool profile = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            if (!ParseClassificationBackend(argv[++i], &backend)) {
                std::cerr << "invalid classification backend\n";
                PrintUsage(demo_name);
                return 1;
            }
        } else if (arg == "--profile") {
            profile = true;
        } else {
            PrintUsage(demo_name);
            return 1;
        }
    }

    if (model_path.empty() || image_path.empty() || top_k <= 0) {
        PrintUsage(demo_name);
        return 1;
    }

    ClassificationRunner runner;
    if (runner.Load(model_path, backend) != 0) {
        std::cerr << "failed to load classification model: " << model_path;
        if (!runner.LastError().empty()) {
            std::cerr << " (" << runner.LastError() << ')';
        }
        std::cerr << '\n';
        return 1;
    }
    runner.SetRuntimeProfilingEnabled(profile);
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';

    ImageNetPreprocessConfig config;
    std::vector<ClassificationResult> results;
    if (runner.Run(image_path, config, top_k, &results) != 0) {
        std::cerr << "failed to run classification model for image: " << image_path;
        if (!runner.LastError().empty()) {
            std::cerr << " (" << runner.LastError() << ')';
        }
        std::cerr << '\n';
        return 1;
    }
    std::cout << "[framework] " << runner.DescribeLastRun() << '\n';
    if (profile) {
        PrintProfile(runner);
    }

    const auto labels = LoadClassificationLabels(labels_path);
    std::cout << std::fixed << std::setprecision(6);
    for (size_t rank = 0; rank < results.size(); ++rank) {
        const auto& result = results[rank];
        std::cout << "rank=" << (rank + 1) << " class_id=" << result.class_id
                  << " probability=" << result.probability << " logit=" << result.logit;
        if (static_cast<size_t>(result.class_id) < labels.size() && !labels[static_cast<size_t>(result.class_id)].empty()) {
            std::cout << " label=" << labels[static_cast<size_t>(result.class_id)];
        }
        std::cout << '\n';
    }
    return 0;
}

}  // namespace demo
}  // namespace feather
