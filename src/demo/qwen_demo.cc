// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "demo/qwen_demo.h"

#include <iostream>
#include <charconv>
#include <limits>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "demo/qwen_chat_template.h"
#include "demo/qwen_chat_session.h"
#include "demo/qwen_runner.h"
#include "demo/qwen_tokenizer.h"

namespace feather {
namespace demo {

namespace {

constexpr const char* kDefaultModel = "models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16.fth";
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
                 " [--system <text>] [--backend host|common|x86|cuda] [--context-size <tokens>] [--max-context <tokens>] [--fp8-weight-cache-mb <MiB>]"
                 " [--max-new-tokens <tokens|-1>] [--profile]\n";
}

bool ParseInteger(const std::string& text, int64_t* value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), *value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
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

int RunTurn(const std::string& user_text, QwenTokenizer* tokenizer, QwenRunner* runner,
            QwenChatSession* session, int max_new_tokens, bool profile) {
    if (tokenizer == nullptr || runner == nullptr || session == nullptr || user_text.empty()) {
        return -1;
    }
    const auto prompt_tokens = tokenizer->Encode(RenderQwenUserTurn(user_text, true));
    if (prompt_tokens.empty()) {
        std::cerr << "failed to encode Qwen prompt\n";
        return -1;
    }
    QwenUtf8Stream utf8_stream;
    bool have_prompt_metrics = false;
    int64_t prompt_tokens_consumed = 0;
    double prompt_prefill_ms = 0.0;
    bool reset_profile_after_prompt = profile;
    const auto on_token = [&](int64_t token_id) {
        if (!have_prompt_metrics) {
            // Turn closure is another Consume call. Retain the actual user
            // prompt metrics before that call replaces the runner's summary.
            have_prompt_metrics = true;
            prompt_tokens_consumed = runner->LastPromptTokensConsumed();
            prompt_prefill_ms = runner->LastPrefillMilliseconds();
        }
        if (reset_profile_after_prompt) {
            // Prompt execution performs one-time immutable-weight packing. Keep
            // the report focused on steady-state autoregressive decode.
            runner->SetRuntimeProfilingEnabled(false);
            runner->SetRuntimeProfilingEnabled(true);
            reset_profile_after_prompt = false;
        }
        if (token_id == tokenizer->EndOfTextTokenId() || token_id == tokenizer->ImEndTokenId()) {
            return;
        }
        const std::string decoded = tokenizer->Decode({token_id});
        const std::string complete = utf8_stream.Push(decoded);
        if (!complete.empty()) {
            std::cout << complete << std::flush;
        }
    };
    QwenGenerationOptions options;
    options.max_new_tokens = max_new_tokens;
    if (session->GenerateTurn(prompt_tokens, options, on_token) != 0) {
        std::cerr << "Qwen generation failed: " << session->LastError() << '\n';
        return -1;
    }
    const std::string pending = utf8_stream.Finish();
    if (!pending.empty()) {
        std::cout << pending << std::flush;
    }
    std::cout << '\n';
    if (have_prompt_metrics) {
        std::cout << "[framework] prompt_tokens=" << prompt_tokens_consumed
                  << " prompt_prefill_ms=" << prompt_prefill_ms
                  << " prefill_chunk=" << runner->PrefillChunkSize() << '\n';
    }
    std::cout << "[framework] " << runner->DescribeLastRun() << " tokens_processed=" << runner->TokensProcessed()
              << " context_capacity=" << runner->MaxContext()
              << " context_growths=" << runner->ContextGrowthCount()
              << " finish_reason=" << QwenFinishReasonName(session->LastFinishReason()) << '\n';
    if (session->LastRebuiltHistory()) {
        std::cout << "[framework] history_rebuilt=true dropped_turns=" << session->LastDroppedTurns()
                  << " retained_turns=" << session->History().TurnCount() << '\n';
    }
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
    int max_new_tokens = 512;
    int64_t requested_context_size = 0;
    int64_t fp8_weight_cache_mb = 256;
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
        } else if ((argument == "--context-size" || argument == "--max-context") && index + 1 < argc) {
            if (!ParseInteger(argv[++index], &requested_context_size) || requested_context_size < 0) {
                std::cerr << "invalid " << argument << ": expected a nonnegative token count\n";
                return 1;
            }
        } else if (argument == "--fp8-weight-cache-mb" && index + 1 < argc) {
            if (!ParseInteger(argv[++index], &fp8_weight_cache_mb) || fp8_weight_cache_mb < 0 ||
                static_cast<uint64_t>(fp8_weight_cache_mb) > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
                std::cerr << "invalid --fp8-weight-cache-mb: expected a nonnegative MiB count\n";
                return 1;
            }
        } else if (argument == "--max-new-tokens" && index + 1 < argc) {
            int64_t parsed_budget = 0;
            if (!ParseInteger(argv[++index], &parsed_budget) ||
                (parsed_budget <= 0 && parsed_budget != -1) ||
                parsed_budget > std::numeric_limits<int>::max()) {
                std::cerr << "invalid --max-new-tokens: expected a positive token count or -1\n";
                return 1;
            }
            max_new_tokens = static_cast<int>(parsed_budget);
        } else if (argument == "--profile") {
            profile = true;
        } else if (argument == "--help") {
            PrintUsage();
            return 0;
        } else {
            std::cerr << "unknown argument or missing value: " << argument << '\n';
            PrintUsage();
            return 1;
        }
    }

    QwenTokenizer tokenizer;
    if (tokenizer.Load(tokenizer_dir) != 0) {
        std::cerr << "failed to load Qwen tokenizer: " << tokenizer.LastError() << '\n';
        return 1;
    }
    QwenRunner runner;
    QwenLoadOptions load_options;
    load_options.context_size = requested_context_size;
    load_options.fp8_weight_cache_bytes = static_cast<size_t>(fp8_weight_cache_mb) * 1024 * 1024;
    if (runner.Load(model_path, backend, load_options) != 0) {
        std::cerr << "failed to load Qwen model: " << runner.LastError() << '\n';
        return 1;
    }
    runner.SetRuntimeProfilingEnabled(profile);
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';

    std::vector<int64_t> system_tokens;
    if (!system_text.empty()) {
        system_tokens = tokenizer.Encode(RenderQwenChat({{"system", system_text}}, false));
        if (system_tokens.empty()) {
            std::cerr << "failed to encode Qwen system prompt\n";
            return 1;
        }
    }
    const auto newline_tokens = tokenizer.Encode("\n");
    if (newline_tokens.empty() || tokenizer.ImEndTokenId() < 0 || tokenizer.EndOfTextTokenId() < 0) {
        std::cerr << "failed to encode Qwen turn closure\n";
        return 1;
    }
    QwenChatSession session(&runner, std::move(system_tokens), tokenizer.ImEndTokenId(), newline_tokens,
                            {tokenizer.EndOfTextTokenId(), tokenizer.ImEndTokenId()});
    if (!prompt.empty()) {
        const int status = RunTurn(prompt, &tokenizer, &runner, &session, max_new_tokens, profile);
        if (status == 0 && profile) {
            PrintRuntimeProfile(runner, 30);
        }
        return status == 0 ? 0 : 1;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (RunTurn(line, &tokenizer, &runner, &session, max_new_tokens, profile) != 0) {
            return 1;
        }
        if (profile) {
            PrintRuntimeProfile(runner, 30);
        }
    }
    return 0;
}

}  // namespace demo
}  // namespace feather
