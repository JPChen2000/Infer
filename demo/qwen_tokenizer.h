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
    bool normalize_nfc_{false};
    std::string split_pattern_;
    std::string last_error_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_TOKENIZER_H
