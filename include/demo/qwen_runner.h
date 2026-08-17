#ifndef FEATHER_DEMO_QWEN_RUNNER_H
#define FEATHER_DEMO_QWEN_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "model/model_io.h"

namespace feather {
namespace demo {

enum class QwenBackend {
    kHost,
    kCommon,
    kX86,
    kCuda,
};

bool ParseQwenBackend(const std::string& value, QwenBackend* backend);
const char* QwenBackendName(QwenBackend backend);

class QwenRunner {
   public:
    int32_t Load(const std::string& model_path, QwenBackend backend = QwenBackend::kCommon);
    int32_t Reset();
    int32_t Consume(const std::vector<int64_t>& token_ids);
    int32_t Generate(const std::vector<int64_t>& prompt_tokens, int max_new_tokens,
                     const std::vector<int64_t>& stop_token_ids, std::vector<int64_t>* generated_tokens);

    int64_t TokensProcessed() const { return tokens_processed_; }
    int64_t MaxContext() const { return max_context_; }
    const std::string& LastError() const { return last_error_; }
    const std::string& DescribeLastBuild() const { return last_build_summary_; }
    const std::string& DescribeLastRun() const { return last_run_summary_; }

   private:
    struct StateBinding {
        std::string input_name;
        std::string output_name;
        int cache_axis{-1};
    };

    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t PrepareExecutableGraph();
    int32_t RunToken(int64_t token_id, int64_t* next_token_id);
    int32_t SetDecodeInputs(int64_t token_id);
    int32_t UpdateStates();
    int32_t CopyState(const StateBinding& binding);
    int64_t SelectGreedyToken() const;

    model::ModelLoader loader_;
    StaticGraph static_graph_;
    RuntimeGraph runtime_graph_;
    GraphLowering lowering_;
    std::vector<StateBinding> states_;
    std::string token_input_name_;
    std::string position_input_name_;
    std::string attention_mask_input_name_;
    std::string logits_output_name_;
    QwenBackend backend_{QwenBackend::kCommon};
    DeviceType backend_device_{DeviceType::COMMON};
    int64_t max_context_{};
    int64_t tokens_processed_{};
    std::string last_error_;
    std::string last_build_summary_;
    std::string last_run_summary_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_RUNNER_H
