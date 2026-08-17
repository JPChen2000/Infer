#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "demo/qwen_chat_template.h"
#include "demo/qwen_tokenizer.h"

namespace {

const std::filesystem::path kTokenizerDir = "models/llm/qwen3.5-0.8b";

void SkipWithoutTokenizerAssets() {
    if (!std::filesystem::is_regular_file(kTokenizerDir / "vocab.json") ||
        !std::filesystem::is_regular_file(kTokenizerDir / "merges.txt")) {
        GTEST_SKIP() << "Qwen tokenizer assets are not present";
    }
}

}  // namespace

TEST(qwen_tokenizer_test, EncodesAndDecodesRealQwenBpeText) {
    SkipWithoutTokenizerAssets();

    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();

    EXPECT_EQ(tokenizer.Encode("Hello, world!"), (std::vector<int64_t>{9419, 11, 1814, 0}));
    EXPECT_EQ(tokenizer.Encode("你好，世界！"), (std::vector<int64_t>{109266, 3709, 96748, 6115}));
    EXPECT_EQ(tokenizer.Decode({109266, 3709, 96748, 6115}), "你好，世界！");
}

TEST(qwen_tokenizer_test, PreservesSpecialTokensAndChatFormatting) {
    SkipWithoutTokenizerAssets();

    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode(" <|im_start|>user\n你好<|im_end|>\n"),
              (std::vector<int64_t>{220, 248045, 846, 198, 109266, 248046, 198}));

    const std::vector<feather::demo::QwenChatMessage> messages = {
        {"system", "You are a helpful assistant."},
        {"user", "你好"},
        {"assistant", "你好！"},
        {"user", "再见"},
    };
    EXPECT_EQ(feather::demo::RenderQwenChat(messages, true),
              "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n你好<|im_end|>\n"
              "<|im_start|>assistant\n你好！<|im_end|>\n"
              "<|im_start|>user\n再见<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

TEST(qwen_tokenizer_test, MatchesOfficialTokenizerForChineseChatPrompt) {
    SkipWithoutTokenizerAssets();

    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();

    const auto prompt = feather::demo::RenderQwenChat(
        {{"system", "You are a helpful assistant."}, {"user", "介绍一下你自己"}}, true);
    EXPECT_EQ(tokenizer.Encode(prompt),
              (std::vector<int64_t>{248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
                                    248045, 846, 198, 113552, 111522, 248046, 198, 248045, 74455, 198,
                                    248068, 271, 248069, 271}));
}
