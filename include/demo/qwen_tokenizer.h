#ifndef FEATHER_DEMO_QWEN_TOKENIZER_H
#define FEATHER_DEMO_QWEN_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace feather {
namespace demo {

class QwenTokenizer {
   public:
    int32_t Load(const std::string& tokenizer_dir);
    std::vector<int64_t> Encode(const std::string& text) const;
    std::string Decode(const std::vector<int64_t>& token_ids) const;

    int64_t TokenId(const std::string& token) const;
    int64_t EndOfTextTokenId() const;
    int64_t ImEndTokenId() const;
    const std::string& LastError() const { return last_error_; }

   private:
    std::vector<std::string> Pretokenize(const std::string& text) const;
    std::vector<std::string> ApplyBpe(const std::string& text) const;
    std::string EncodeBytes(const std::string& text) const;
    std::string DecodeBytes(const std::string& text) const;

    std::unordered_map<std::string, int64_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::unordered_map<std::string, int64_t> special_token_to_id_;
    std::unordered_map<int64_t, std::string> special_id_to_token_;
    std::vector<std::string> special_tokens_by_length_;
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    std::vector<std::string> byte_encoder_;
    std::string last_error_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_TOKENIZER_H
