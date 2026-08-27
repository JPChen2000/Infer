#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "demo/qwen_chat_template.h"
#include "demo/qwen_tokenizer.h"

namespace {

const std::filesystem::path kTokenizerDir =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "models" / "llm" / "qwen3.5-0.8b";

bool HasTokenizerAssets(const std::filesystem::path& tokenizer_dir) {
    return std::filesystem::is_regular_file(tokenizer_dir / "vocab.json") &&
           std::filesystem::is_regular_file(tokenizer_dir / "merges.txt") &&
           std::filesystem::is_regular_file(tokenizer_dir / "tokenizer.json");
}

class TokenizerAssetFixture {
   public:
    TokenizerAssetFixture()
        : directory_(std::filesystem::temp_directory_path() /
                     ("feather_qwen_tokenizer_assets_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}

    ~TokenizerAssetFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    bool CreateFile(const std::string& name) const {
        std::error_code error;
        std::filesystem::create_directories(directory_, error);
        if (error) {
            return false;
        }
        std::ofstream output(directory_ / name, std::ios::binary | std::ios::trunc);
        return output.good();
    }

    const std::filesystem::path& directory() const { return directory_; }

   private:
    std::filesystem::path directory_;
};

class ScopedWorkingDirectory {
   public:
    explicit ScopedWorkingDirectory(const std::filesystem::path& directory) : original_(std::filesystem::current_path()) {
        std::error_code error;
        std::filesystem::current_path(directory, error);
        changed_ = !error;
    }

    ~ScopedWorkingDirectory() {
        std::error_code error;
        std::filesystem::current_path(original_, error);
    }

    bool changed() const { return changed_; }

   private:
    std::filesystem::path original_;
    bool changed_{};
};

}  // namespace

TEST(qwen_tokenizer_test, RejectsIncompleteTokenizerAssetSet) {
    TokenizerAssetFixture fixture;
    ASSERT_TRUE(fixture.CreateFile("vocab.json"));
    ASSERT_TRUE(fixture.CreateFile("merges.txt"));

    EXPECT_FALSE(HasTokenizerAssets(fixture.directory()));
}

TEST(qwen_tokenizer_test, ResolvesAssetsOutsideRepositoryWorkingDirectory) {
    const auto source_tokenizer_dir =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "models" / "llm" / "qwen3.5-0.8b";
    if (!HasTokenizerAssets(source_tokenizer_dir)) {
        GTEST_SKIP() << "Qwen tokenizer assets are not present";
    }

    ScopedWorkingDirectory working_directory(std::filesystem::path(__FILE__).parent_path());
    ASSERT_TRUE(working_directory.changed());
    EXPECT_TRUE(HasTokenizerAssets(kTokenizerDir));
}

TEST(qwen_tokenizer_test, EncodesAndDecodesRealQwenBpeText) {
    if (!HasTokenizerAssets(kTokenizerDir)) {
        GTEST_SKIP() << "Qwen tokenizer assets are not present";
    }

    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();

    EXPECT_EQ(tokenizer.Encode("Hello, world!"), (std::vector<int64_t>{9419, 11, 1814, 0}));
    EXPECT_EQ(tokenizer.Encode("你好，世界！"), (std::vector<int64_t>{109266, 3709, 96748, 6115}));
    EXPECT_EQ(tokenizer.Decode({109266, 3709, 96748, 6115}), "你好，世界！");
}

TEST(qwen_tokenizer_test, PreservesSpecialTokensAndChatFormatting) {
    if (!HasTokenizerAssets(kTokenizerDir)) {
        GTEST_SKIP() << "Qwen tokenizer assets are not present";
    }

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
    if (!HasTokenizerAssets(kTokenizerDir)) {
        GTEST_SKIP() << "Qwen tokenizer assets are not present";
    }

    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();

    const auto prompt = feather::demo::RenderQwenChat(
        {{"system", "You are a helpful assistant."}, {"user", "介绍一下你自己"}}, true);
    EXPECT_EQ(tokenizer.Encode(prompt),
              (std::vector<int64_t>{248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
                                    248045, 846, 198, 113552, 111522, 248046, 198, 248045, 74455, 198,
                                    248068, 271, 248069, 271}));
}
