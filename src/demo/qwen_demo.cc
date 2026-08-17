#include "demo/qwen_demo.h"

#include <iostream>
#include <string>
#include <vector>

#include "demo/qwen_chat_template.h"
#include "demo/qwen_runner.h"
#include "demo/qwen_tokenizer.h"

namespace feather {
namespace demo {

namespace {

constexpr const char* kDefaultModel = "models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx128.fth";
constexpr const char* kDefaultTokenizer = "models/llm/qwen3.5-0.8b";

void PrintUsage() {
    std::cerr << "usage: qwen_demo [--model <model.fth>] [--tokenizer <directory>] [--prompt <text>]"
                 " [--system <text>] [--backend host|common|x86|cuda] [--max-context <tokens>]"
                 " [--max-new-tokens <tokens>]\n";
}

int RunTurn(const std::string& user_text, bool first_turn, const std::string& system_text, QwenTokenizer* tokenizer,
            QwenRunner* runner, int max_new_tokens, std::vector<QwenChatMessage>* history) {
    if (tokenizer == nullptr || runner == nullptr || history == nullptr || user_text.empty()) {
        return -1;
    }
    std::string formatted;
    if (first_turn && !system_text.empty()) {
        formatted = RenderQwenChat({{"system", system_text}, {"user", user_text}}, true);
    } else {
        formatted = RenderQwenUserTurn(user_text, true);
    }
    const auto prompt_tokens = tokenizer->Encode(formatted);
    if (prompt_tokens.empty()) {
        std::cerr << "failed to encode Qwen prompt\n";
        return -1;
    }
    std::vector<int64_t> generated;
    const std::vector<int64_t> stop_ids = {tokenizer->EndOfTextTokenId(), tokenizer->ImEndTokenId()};
    if (runner->Generate(prompt_tokens, max_new_tokens, stop_ids, &generated) != 0) {
        std::cerr << "Qwen generation failed: " << runner->LastError() << '\n';
        return -1;
    }
    const bool ended_with_stop = !generated.empty() &&
                                 (generated.back() == tokenizer->EndOfTextTokenId() ||
                                  generated.back() == tokenizer->ImEndTokenId());
    if (!ended_with_stop && runner->Consume({tokenizer->ImEndTokenId()}) != 0) {
        std::cerr << "failed to close Qwen assistant turn: " << runner->LastError() << '\n';
        return -1;
    }
    std::vector<int64_t> response_tokens;
    response_tokens.reserve(generated.size());
    for (const auto token : generated) {
        if (token != tokenizer->EndOfTextTokenId() && token != tokenizer->ImEndTokenId()) {
            response_tokens.push_back(token);
        }
    }
    const std::string response = tokenizer->Decode(response_tokens);
    std::cout << response << '\n';
    history->push_back({"user", user_text});
    history->push_back({"assistant", response});
    std::cout << "[framework] " << runner->DescribeLastRun() << " tokens_processed=" << runner->TokensProcessed()
              << '\n';
    return 0;
}

}  // namespace

int RunQwenDemo(int argc, char** argv) {
    std::string model_path = kDefaultModel;
    std::string tokenizer_dir = kDefaultTokenizer;
    std::string prompt;
    std::string system_text = "You are a helpful assistant.";
    QwenBackend backend = QwenBackend::kCommon;
    int max_new_tokens = 64;
    int requested_max_context = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            model_path = argv[++index];
        } else if (argument == "--tokenizer" && index + 1 < argc) {
            tokenizer_dir = argv[++index];
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else if (argument == "--system" && index + 1 < argc) {
            system_text = argv[++index];
        } else if (argument == "--backend" && index + 1 < argc) {
            if (!ParseQwenBackend(argv[++index], &backend)) {
                std::cerr << "invalid Qwen backend\n";
                PrintUsage();
                return 1;
            }
        } else if (argument == "--max-context" && index + 1 < argc) {
            requested_max_context = std::stoi(argv[++index]);
        } else if (argument == "--max-new-tokens" && index + 1 < argc) {
            max_new_tokens = std::stoi(argv[++index]);
        } else if (argument == "--help") {
            PrintUsage();
            return 0;
        } else {
            PrintUsage();
            return 1;
        }
    }
    if (max_new_tokens <= 0 || requested_max_context < 0) {
        PrintUsage();
        return 1;
    }

    QwenTokenizer tokenizer;
    if (tokenizer.Load(tokenizer_dir) != 0) {
        std::cerr << "failed to load Qwen tokenizer: " << tokenizer.LastError() << '\n';
        return 1;
    }
    QwenRunner runner;
    if (runner.Load(model_path, backend) != 0) {
        std::cerr << "failed to load Qwen model: " << runner.LastError() << '\n';
        return 1;
    }
    if (requested_max_context != 0 && requested_max_context != runner.MaxContext()) {
        std::cerr << "requested max context does not match the fixed FTH graph context " << runner.MaxContext() << '\n';
        return 1;
    }
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';

    std::vector<QwenChatMessage> history;
    if (!prompt.empty()) {
        return RunTurn(prompt, true, system_text, &tokenizer, &runner, max_new_tokens, &history) == 0 ? 0 : 1;
    }
    std::string line;
    bool first_turn = true;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (RunTurn(line, first_turn, system_text, &tokenizer, &runner, max_new_tokens, &history) != 0) {
            return 1;
        }
        first_turn = false;
    }
    return 0;
}

}  // namespace demo
}  // namespace feather
