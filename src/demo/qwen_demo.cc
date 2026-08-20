#include "demo/qwen_demo.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "demo/qwen_chat_template.h"
#include "demo/qwen_runner.h"
#include "demo/qwen_tokenizer.h"

namespace feather {
namespace demo {

namespace {

constexpr const char* kDefaultModel = "models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx128.fth";
constexpr const char* kDefaultTokenizer = "models/llm/qwen3.5-0.8b";

bool IsUtf8ContinuationByte(unsigned char byte) { return byte >= 0x80 && byte <= 0xbf; }

constexpr const char* kUtf8Replacement = "\xEF\xBF\xBD";

bool IsValidUtf8Sequence(const std::string& text, size_t cursor, size_t length) {
    const auto first = static_cast<unsigned char>(text[cursor]);
    if (length == 2) {
        return IsUtf8ContinuationByte(static_cast<unsigned char>(text[cursor + 1]));
    }
    if (length == 3) {
        const auto second = static_cast<unsigned char>(text[cursor + 1]);
        return IsUtf8ContinuationByte(second) &&
               IsUtf8ContinuationByte(static_cast<unsigned char>(text[cursor + 2])) &&
               (first != 0xe0 || second >= 0xa0) && (first != 0xed || second <= 0x9f);
    }
    const auto second = static_cast<unsigned char>(text[cursor + 1]);
    return IsUtf8ContinuationByte(second) &&
           IsUtf8ContinuationByte(static_cast<unsigned char>(text[cursor + 2])) &&
           IsUtf8ContinuationByte(static_cast<unsigned char>(text[cursor + 3])) &&
           (first != 0xf0 || second >= 0x90) && (first != 0xf4 || second <= 0x8f);
}

std::string ConsumeUtf8(std::string* pending, bool flush_incomplete) {
    if (pending == nullptr) {
        return "";
    }
    std::string complete;
    size_t cursor = 0;
    while (cursor < pending->size()) {
        const auto first = static_cast<unsigned char>((*pending)[cursor]);
        size_t length = 1;
        if (first <= 0x7f) {
            complete.push_back(static_cast<char>(first));
            ++cursor;
            continue;
        } else if (first >= 0xc2 && first <= 0xdf) {
            length = 2;
        } else if (first >= 0xe0 && first <= 0xef) {
            length = 3;
        } else if (first >= 0xf0 && first <= 0xf4) {
            length = 4;
        } else {
            complete += kUtf8Replacement;
            ++cursor;
            continue;
        }
        const size_t available = pending->size() - cursor;
        if (available < length) {
            bool valid_prefix = true;
            for (size_t offset = 1; offset < available; ++offset) {
                if (!IsUtf8ContinuationByte(static_cast<unsigned char>((*pending)[cursor + offset]))) {
                    valid_prefix = false;
                    break;
                }
            }
            if (valid_prefix && !flush_incomplete) {
                break;
            }
            complete += kUtf8Replacement;
            ++cursor;
            continue;
        }
        if (!IsValidUtf8Sequence(*pending, cursor, length)) {
            complete += kUtf8Replacement;
            ++cursor;
            continue;
        }
        complete.append(*pending, cursor, length);
        cursor += length;
    }
    pending->erase(0, cursor);
    return complete;
}

void PrintUsage() {
    std::cerr << "usage: qwen_demo [--model <model.fth>] [--tokenizer <directory>] [--prompt <text>]"
                 " [--system <text>] [--backend host|common|x86|cuda] [--max-context <tokens>]"
                 " [--max-new-tokens <tokens>] [--profile]\n";
}

void PrintRuntimeProfile(const QwenRunner& runner, size_t limit) {
    auto summaries = runner.RuntimeProfileSummaries();
    std::sort(summaries.begin(), summaries.end(),
              [](const RuntimeProfileSummary& lhs, const RuntimeProfileSummary& rhs) {
                  return lhs.total_ms > rhs.total_ms;
              });
    const size_t count = std::min(limit, summaries.size());
    std::cout << "[profile] top_runtime_nodes=" << count << '\n';
    for (size_t index = 0; index < count; ++index) {
        const auto& item = summaries[index];
        std::cout << "[profile] #" << (index + 1)
                  << " node=" << item.node_name
                  << " op=" << item.op_type
                  << " calls=" << item.call_count
                  << " total_ms=" << item.total_ms
                  << " avg_ms=" << item.avg_ms
                  << " min_ms=" << item.min_ms
                  << " max_ms=" << item.max_ms << '\n';
    }

    std::unordered_map<std::string, RuntimeProfileSummary> by_op;
    for (const auto& item : summaries) {
        auto& aggregate = by_op[item.op_type];
        aggregate.op_type = item.op_type;
        aggregate.call_count += item.call_count;
        aggregate.total_ms += item.total_ms;
    }
    std::vector<RuntimeProfileSummary> op_summaries;
    op_summaries.reserve(by_op.size());
    for (auto& item : by_op) {
        item.second.avg_ms = item.second.call_count == 0
                                 ? 0.0
                                 : item.second.total_ms / static_cast<double>(item.second.call_count);
        op_summaries.push_back(std::move(item.second));
    }
    std::sort(op_summaries.begin(), op_summaries.end(),
              [](const RuntimeProfileSummary& lhs, const RuntimeProfileSummary& rhs) {
                  return lhs.total_ms > rhs.total_ms;
              });
    std::cout << "[profile] op_totals=" << op_summaries.size() << '\n';
    for (const auto& item : op_summaries) {
        std::cout << "[profile] op=" << item.op_type
                  << " calls=" << item.call_count
                  << " total_ms=" << item.total_ms
                  << " avg_ms=" << item.avg_ms << '\n';
    }
}

int RunTurn(const std::string& user_text, bool first_turn, const std::string& system_text, QwenTokenizer* tokenizer,
            QwenRunner* runner, int max_new_tokens, bool profile, std::vector<QwenChatMessage>* history) {
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
    const std::vector<int64_t> stop_ids = {tokenizer->EndOfTextTokenId(), tokenizer->ImEndTokenId()};
    std::vector<int64_t> generated;
    QwenUtf8Stream utf8_stream;
    bool reset_profile_after_prompt = profile;
    const auto on_token = [&](int64_t token_id) {
        if (reset_profile_after_prompt) {
            // Prompt execution performs one-time immutable-weight packing. Keep
            // the report focused on steady-state autoregressive decode.
            runner->SetRuntimeProfilingEnabled(false);
            runner->SetRuntimeProfilingEnabled(true);
            reset_profile_after_prompt = false;
        }
        generated.push_back(token_id);
        if (token_id == tokenizer->EndOfTextTokenId() || token_id == tokenizer->ImEndTokenId()) {
            return;
        }
        const std::string decoded = tokenizer->Decode({token_id});
        const std::string complete = utf8_stream.Push(decoded);
        if (!complete.empty()) {
            std::cout << complete << std::flush;
        }
    };
    if (runner->GenerateStream(prompt_tokens, max_new_tokens, stop_ids, on_token) != 0) {
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
    const std::string pending = utf8_stream.Finish();
    if (!pending.empty()) {
        std::cout << pending << std::flush;
    }
    const std::string response = tokenizer->Decode(response_tokens);
    std::cout << '\n';
    history->push_back({"user", user_text});
    history->push_back({"assistant", response});
    std::cout << "[framework] " << runner->DescribeLastRun() << " tokens_processed=" << runner->TokensProcessed()
              << '\n';
    return 0;
}

}  // namespace

std::string QwenUtf8Stream::Push(const std::string& chunk) {
    pending_ += chunk;
    return ConsumeUtf8(&pending_, false);
}

std::string QwenUtf8Stream::Finish() {
    return ConsumeUtf8(&pending_, true);
}

int RunQwenDemo(int argc, char** argv) {
    std::string model_path = kDefaultModel;
    std::string tokenizer_dir = kDefaultTokenizer;
    std::string prompt;
    std::string system_text = "You are a helpful assistant.";
    QwenBackend backend = QwenBackend::kX86;
    int max_new_tokens = 64;
    int requested_max_context = 0;
    bool profile = false;
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
        } else if (argument == "--profile") {
            profile = true;
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
    runner.SetRuntimeProfilingEnabled(profile);
    if (requested_max_context != 0 && requested_max_context != runner.MaxContext()) {
        std::cerr << "requested max context does not match the fixed FTH graph context " << runner.MaxContext() << '\n';
        return 1;
    }
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';

    std::vector<QwenChatMessage> history;
    if (!prompt.empty()) {
        const int status = RunTurn(prompt, true, system_text, &tokenizer, &runner, max_new_tokens, profile, &history);
        if (status == 0 && profile) {
            PrintRuntimeProfile(runner, 30);
        }
        return status == 0 ? 0 : 1;
    }
    std::string line;
    bool first_turn = true;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (RunTurn(line, first_turn, system_text, &tokenizer, &runner, max_new_tokens, profile, &history) != 0) {
            return 1;
        }
        if (profile) {
            PrintRuntimeProfile(runner, 30);
        }
        first_turn = false;
    }
    return 0;
}

}  // namespace demo
}  // namespace feather
