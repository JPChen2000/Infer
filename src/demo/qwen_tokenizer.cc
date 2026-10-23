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

#include "demo/qwen_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>

#ifdef FEATHER_WITH_ICU
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#endif

namespace feather {
namespace demo {

namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        return "";
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void AppendUtf8(uint32_t codepoint, std::string* output) {
    if (output == nullptr || codepoint > 0x10FFFF) {
        return;
    }
    if (codepoint <= 0x7F) {
        output->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        output->push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        output->push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        output->push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

bool ReadUtf8Codepoint(const std::string& text, size_t offset, uint32_t* codepoint, size_t* width) {
    if (codepoint == nullptr || width == nullptr || offset >= text.size()) {
        return false;
    }
    const auto first = static_cast<uint8_t>(text[offset]);
    if (first < 0x80) {
        *codepoint = first;
        *width = 1;
        return true;
    }
    size_t expected_width = 0;
    uint32_t value = 0;
    if ((first & 0xE0) == 0xC0) {
        expected_width = 2;
        value = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
        expected_width = 3;
        value = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
        expected_width = 4;
        value = first & 0x07;
    } else {
        *codepoint = first;
        *width = 1;
        return true;
    }
    if (offset + expected_width > text.size()) {
        *codepoint = first;
        *width = 1;
        return true;
    }
    for (size_t index = 1; index < expected_width; ++index) {
        const auto next = static_cast<uint8_t>(text[offset + index]);
        if ((next & 0xC0) != 0x80) {
            *codepoint = first;
            *width = 1;
            return true;
        }
        value = (value << 6) | (next & 0x3F);
    }
    *codepoint = value;
    *width = expected_width;
    return true;
}

class JsonCursor {
   public:
    explicit JsonCursor(std::string_view text) : text_(text) {}

    void SkipWhitespace() {
        while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) {
            ++position_;
        }
    }

    bool Consume(char expected) {
        SkipWhitespace();
        if (position_ >= text_.size() || text_[position_] != expected) {
            return false;
        }
        ++position_;
        return true;
    }

    bool ParseString(std::string* value) {
        if (value == nullptr || !Consume('"')) {
            return false;
        }
        value->clear();
        while (position_ < text_.size()) {
            const char current = text_[position_++];
            if (current == '"') {
                return true;
            }
            if (current != '\\') {
                value->push_back(current);
                continue;
            }
            if (position_ >= text_.size()) {
                return false;
            }
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"':
                    value->push_back('"');
                    break;
                case '\\':
                    value->push_back('\\');
                    break;
                case '/':
                    value->push_back('/');
                    break;
                case 'b':
                    value->push_back('\b');
                    break;
                case 'f':
                    value->push_back('\f');
                    break;
                case 'n':
                    value->push_back('\n');
                    break;
                case 'r':
                    value->push_back('\r');
                    break;
                case 't':
                    value->push_back('\t');
                    break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if (!ParseHexCodepoint(&codepoint)) {
                        return false;
                    }
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF && position_ + 2 <= text_.size() &&
                        text_[position_] == '\\' && text_[position_ + 1] == 'u') {
                        position_ += 2;
                        uint32_t low = 0;
                        if (!ParseHexCodepoint(&low) || low < 0xDC00 || low > 0xDFFF) {
                            return false;
                        }
                        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                    }
                    AppendUtf8(codepoint, value);
                    break;
                }
                default:
                    return false;
            }
        }
        return false;
    }

    bool ParseInt(int64_t* value) {
        if (value == nullptr) {
            return false;
        }
        SkipWhitespace();
        const size_t begin = position_;
        if (position_ < text_.size() && text_[position_] == '-') {
            ++position_;
        }
        const size_t digit_begin = position_;
        while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) {
            ++position_;
        }
        if (digit_begin == position_) {
            return false;
        }
        try {
            *value = std::stoll(std::string(text_.substr(begin, position_ - begin)));
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    bool ConsumeNull() {
        SkipWhitespace();
        if (text_.substr(position_, 4) != "null") return false;
        position_ += 4;
        return true;
    }

    bool SkipValue() {
        SkipWhitespace();
        if (position_ >= text_.size()) {
            return false;
        }
        if (text_[position_] == '"') {
            std::string ignored;
            return ParseString(&ignored);
        }
        if (text_[position_] == '{') {
            ++position_;
            SkipWhitespace();
            if (position_ < text_.size() && text_[position_] == '}') {
                ++position_;
                return true;
            }
            while (true) {
                std::string key;
                if (!ParseString(&key) || !Consume(':') || !SkipValue()) {
                    return false;
                }
                if (Consume('}')) {
                    return true;
                }
                if (!Consume(',')) {
                    return false;
                }
            }
        }
        if (text_[position_] == '[') {
            ++position_;
            SkipWhitespace();
            if (position_ < text_.size() && text_[position_] == ']') {
                ++position_;
                return true;
            }
            while (true) {
                if (!SkipValue()) {
                    return false;
                }
                if (Consume(']')) {
                    return true;
                }
                if (!Consume(',')) {
                    return false;
                }
            }
        }
        while (position_ < text_.size() && text_[position_] != ',' && text_[position_] != ']' &&
               text_[position_] != '}' && !std::isspace(static_cast<unsigned char>(text_[position_]))) {
            ++position_;
        }
        return position_ > 0;
    }

   private:
    bool ParseHexCodepoint(uint32_t* codepoint) {
        if (codepoint == nullptr || position_ + 4 > text_.size()) {
            return false;
        }
        *codepoint = 0;
        for (size_t index = 0; index < 4; ++index) {
            const char value = text_[position_++];
            *codepoint <<= 4;
            if (value >= '0' && value <= '9') {
                *codepoint |= static_cast<uint32_t>(value - '0');
            } else if (value >= 'a' && value <= 'f') {
                *codepoint |= static_cast<uint32_t>(value - 'a' + 10);
            } else if (value >= 'A' && value <= 'F') {
                *codepoint |= static_cast<uint32_t>(value - 'A' + 10);
            } else {
                return false;
            }
        }
        return true;
    }

    std::string_view text_;
    size_t position_{};
};

bool ParseVocab(const std::string& text, std::unordered_map<std::string, int64_t>* vocab,
                std::vector<std::string>* id_to_token) {
    if (vocab == nullptr || id_to_token == nullptr) {
        return false;
    }
    JsonCursor cursor(text);
    if (!cursor.Consume('{')) {
        return false;
    }
    while (true) {
        cursor.SkipWhitespace();
        if (cursor.Consume('}')) {
            return true;
        }
        std::string token;
        int64_t id = -1;
        if (!cursor.ParseString(&token) || !cursor.Consume(':') || !cursor.ParseInt(&id) || id < 0) {
            return false;
        }
        (*vocab)[token] = id;
        if (static_cast<size_t>(id) >= id_to_token->size()) {
            id_to_token->resize(static_cast<size_t>(id) + 1);
        }
        (*id_to_token)[static_cast<size_t>(id)] = token;
        if (cursor.Consume('}')) {
            return true;
        }
        if (!cursor.Consume(',')) {
            return false;
        }
    }
}

bool ParseAddedTokenObject(JsonCursor* cursor, int64_t* id, std::string* content) {
    if (cursor == nullptr || id == nullptr || content == nullptr || !cursor->Consume('{')) {
        return false;
    }
    *id = -1;
    content->clear();
    while (true) {
        cursor->SkipWhitespace();
        if (cursor->Consume('}')) {
            return *id >= 0 && !content->empty();
        }
        std::string key;
        if (!cursor->ParseString(&key) || !cursor->Consume(':')) {
            return false;
        }
        if (key == "id") {
            if (!cursor->ParseInt(id)) {
                return false;
            }
        } else if (key == "content") {
            if (!cursor->ParseString(content)) {
                return false;
            }
        } else if (!cursor->SkipValue()) {
            return false;
        }
        if (cursor->Consume('}')) {
            return *id >= 0 && !content->empty();
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }
}

bool ParseSpecialTokens(const std::string& text, std::unordered_map<std::string, int64_t>* token_to_id,
                        std::unordered_map<int64_t, std::string>* id_to_token) {
    if (token_to_id == nullptr || id_to_token == nullptr) {
        return false;
    }
    const size_t key = text.find("\"added_tokens\"");
    if (key == std::string::npos) {
        return false;
    }
    const size_t colon = text.find(':', key);
    if (colon == std::string::npos) {
        return false;
    }
    JsonCursor cursor(std::string_view(text).substr(colon + 1));
    if (!cursor.Consume('[')) {
        return false;
    }
    while (true) {
        cursor.SkipWhitespace();
        if (cursor.Consume(']')) {
            return true;
        }
        int64_t id = -1;
        std::string content;
        if (!ParseAddedTokenObject(&cursor, &id, &content)) {
            return false;
        }
        (*token_to_id)[content] = id;
        (*id_to_token)[id] = content;
        if (cursor.Consume(']')) {
            return true;
        }
        if (!cursor.Consume(',')) {
            return false;
        }
    }
}

bool IsNewline(uint32_t codepoint) { return codepoint == '\n' || codepoint == '\r'; }

bool IsWhitespace(uint32_t codepoint) {
#ifdef FEATHER_WITH_ICU
    return u_isUWhiteSpace(static_cast<UChar32>(codepoint));
#else
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\v' || codepoint == '\f' || IsNewline(codepoint) ||
           codepoint == 0x3000;
#endif
}

bool IsDigit(uint32_t codepoint) {
#ifdef FEATHER_WITH_ICU
    const auto category = u_charType(static_cast<UChar32>(codepoint));
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER || category == U_OTHER_NUMBER;
#else
    return (codepoint >= '0' && codepoint <= '9') || (codepoint >= 0xFF10 && codepoint <= 0xFF19);
#endif
}

#ifndef FEATHER_WITH_ICU
bool IsPunctuation(uint32_t codepoint) {
    return (codepoint >= 0x3001 && codepoint <= 0x303F) || (codepoint >= 0xFF00 && codepoint <= 0xFF0F) ||
           (codepoint >= 0xFF1A && codepoint <= 0xFF20) || (codepoint >= 0xFF3B && codepoint <= 0xFF40) ||
           (codepoint >= 0xFF5B && codepoint <= 0xFF65) || codepoint >= 0x1F000;
}
#endif

bool IsLetter(uint32_t codepoint) {
#ifdef FEATHER_WITH_ICU
    const auto category = u_charType(static_cast<UChar32>(codepoint));
    return category >= U_UPPERCASE_LETTER && category <= U_OTHER_LETTER;
#else
    if ((codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z') ||
        (codepoint >= 0xFF21 && codepoint <= 0xFF3A) || (codepoint >= 0xFF41 && codepoint <= 0xFF5A)) {
        return true;
    }
    return codepoint >= 0x80 && !IsWhitespace(codepoint) && !IsDigit(codepoint) && !IsPunctuation(codepoint);
#endif
}

bool ParseNormalizer(const std::string& text, bool* nfc) {
    *nfc = false;
    JsonCursor cursor(text);
    if (!cursor.Consume('{')) return false;
    while (!cursor.Consume('}')) {
        std::string key;
        if (!cursor.ParseString(&key) || !cursor.Consume(':')) return false;
        if (key != "normalizer") {
            if (!cursor.SkipValue()) return false;
        } else if (!cursor.ConsumeNull()) {
            std::string type, field;
            if (!cursor.Consume('{') || !cursor.ParseString(&field) || field != "type" || !cursor.Consume(':') ||
                !cursor.ParseString(&type) || type != "NFC" || !cursor.Consume('}'))
                return false;
            *nfc = true;
        }
        if (cursor.Consume('}')) return true;
        if (!cursor.Consume(',')) return false;
    }
    return true;
}

bool FindRegex(JsonCursor* cursor, std::string* pattern) {
    if (cursor->Consume('{')) {
        while (!cursor->Consume('}')) {
            std::string key;
            if (!cursor->ParseString(&key) || !cursor->Consume(':')) return false;
            if (key == "Regex") {
                if (!pattern->empty() || !cursor->ParseString(pattern)) return false;
            } else if (!FindRegex(cursor, pattern)) {
                return false;
            }
            if (cursor->Consume('}')) return true;
            if (!cursor->Consume(',')) return false;
        }
        return true;
    }
    if (cursor->Consume('[')) {
        while (!cursor->Consume(']')) {
            if (!FindRegex(cursor, pattern)) return false;
            if (cursor->Consume(']')) return true;
            if (!cursor->Consume(',')) return false;
        }
        return true;
    }
    return cursor->SkipValue();
}

bool ParseSplitPattern(const std::string& text, std::string* pattern) {
    pattern->clear();
    JsonCursor cursor(text);
    if (!cursor.Consume('{')) return false;
    while (!cursor.Consume('}')) {
        std::string key;
        if (!cursor.ParseString(&key) || !cursor.Consume(':')) return false;
        if (key == "pre_tokenizer") {
            if (!FindRegex(&cursor, pattern)) return false;
        } else if (!cursor.SkipValue()) {
            return false;
        }
        if (cursor.Consume('}')) return true;
        if (!cursor.Consume(',')) return false;
    }
    return true;
}

#ifdef FEATHER_WITH_ICU
bool NormalizeNfc(const std::string& text, std::string* normalized) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return false;
    UErrorCode error = U_ZERO_ERROR;
    int32_t size = 0;
    u_strFromUTF8(nullptr, 0, &size, text.data(), static_cast<int32_t>(text.size()), &error);
    if (error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) return false;
    error = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<size_t>(size) + 1);
    u_strFromUTF8(utf16.data(), static_cast<int32_t>(utf16.size()), &size, text.data(),
                  static_cast<int32_t>(text.size()), &error);
    if (U_FAILURE(error)) return false;
    const auto* nfc = unorm2_getNFCInstance(&error);
    if (U_FAILURE(error)) return false;
    error = U_ZERO_ERROR;
    int32_t count = unorm2_normalize(nfc, utf16.data(), size, nullptr, 0, &error);
    if (error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) return false;
    error = U_ZERO_ERROR;
    std::vector<UChar> result(static_cast<size_t>(count) + 1);
    count = unorm2_normalize(nfc, utf16.data(), size, result.data(), static_cast<int32_t>(result.size()), &error);
    if (U_FAILURE(error)) return false;
    int32_t bytes = 0;
    error = U_ZERO_ERROR;
    u_strToUTF8(nullptr, 0, &bytes, result.data(), count, &error);
    if (error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) return false;
    error = U_ZERO_ERROR;
    normalized->resize(bytes);
    u_strToUTF8(normalized->data(), bytes, &bytes, result.data(), count, &error);
    return U_SUCCESS(error);
}
#endif

struct Utf8Unit {
    size_t begin{};
    size_t end{};
    uint32_t codepoint{};
};

std::vector<Utf8Unit> SplitUtf8(const std::string& text) {
    std::vector<Utf8Unit> units;
    for (size_t offset = 0; offset < text.size();) {
        uint32_t codepoint = 0;
        size_t width = 0;
        if (!ReadUtf8Codepoint(text, offset, &codepoint, &width)) {
            break;
        }
        units.push_back({offset, offset + width, codepoint});
        offset += width;
    }
    return units;
}

std::string PairKey(const std::string& lhs, const std::string& rhs) { return lhs + '\0' + rhs; }

}  // namespace

int32_t QwenTokenizer::Load(const std::string& tokenizer_dir) {
    last_error_.clear();
    normalize_nfc_ = false;
    split_pattern_.clear();
    token_to_id_.clear();
    id_to_token_.clear();
    merge_ranks_.clear();
    special_token_to_id_.clear();
    special_id_to_token_.clear();
    special_tokens_by_length_.clear();
    byte_decoder_.clear();
    byte_encoder_.clear();

    const std::string vocab_text = ReadFile(tokenizer_dir + "/vocab.json");
    const std::string merges_text = ReadFile(tokenizer_dir + "/merges.txt");
    const std::string tokenizer_text = ReadFile(tokenizer_dir + "/tokenizer.json");
    if (vocab_text.empty() || merges_text.empty() || tokenizer_text.empty()) {
        last_error_ = "missing Qwen tokenizer files under " + tokenizer_dir;
        return -1;
    }
    if (!ParseNormalizer(tokenizer_text, &normalize_nfc_)) {
        last_error_ = "unsupported or malformed Qwen tokenizer normalizer (expected NFC or null)";
        return -1;
    }
    if (!ParseSplitPattern(tokenizer_text, &split_pattern_)) {
        last_error_ = "malformed Qwen tokenizer pre-tokenizer regex";
        return -1;
    }
#ifndef FEATHER_WITH_ICU
    if (normalize_nfc_ || !split_pattern_.empty()) {
        last_error_ = "Qwen tokenizer requires ICU support for Unicode normalization and splitting";
        return -1;
    }
#else
    if (!split_pattern_.empty()) {
        UErrorCode error = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexPattern> pattern(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(split_pattern_), 0, error));
        if (U_FAILURE(error) || !pattern) {
            last_error_ = "unsupported Qwen tokenizer pre-tokenizer regex";
            return -1;
        }
    }
#endif
    if (!ParseVocab(vocab_text, &token_to_id_, &id_to_token_) ||
        !ParseSpecialTokens(tokenizer_text, &special_token_to_id_, &special_id_to_token_)) {
        last_error_ = "failed to parse Qwen tokenizer JSON";
        return -1;
    }
    for (const auto& item : special_token_to_id_) {
        token_to_id_[item.first] = item.second;
        special_tokens_by_length_.push_back(item.first);
    }
    std::sort(special_tokens_by_length_.begin(), special_tokens_by_length_.end(),
              [](const std::string& lhs, const std::string& rhs) { return lhs.size() > rhs.size(); });

    std::istringstream merges_input(merges_text);
    std::string line;
    int32_t rank = 0;
    while (std::getline(merges_input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t separator = line.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= line.size()) {
            last_error_ = "invalid Qwen BPE merge entry";
            return -1;
        }
        merge_ranks_.emplace(PairKey(line.substr(0, separator), line.substr(separator + 1)), rank++);
    }
    if (merge_ranks_.empty()) {
        last_error_ = "Qwen BPE merges are empty";
        return -1;
    }

    std::vector<bool> present(256, false);
    for (int value = 33; value <= 126; ++value) present[static_cast<size_t>(value)] = true;
    for (int value = 161; value <= 172; ++value) present[static_cast<size_t>(value)] = true;
    for (int value = 174; value <= 255; ++value) present[static_cast<size_t>(value)] = true;
    byte_encoder_.resize(256);
    uint32_t next_codepoint = 256;
    for (int byte = 0; byte <= 255; ++byte) {
        uint32_t codepoint = static_cast<uint32_t>(byte);
        if (!present[static_cast<size_t>(byte)]) {
            codepoint = next_codepoint++;
        }
        AppendUtf8(codepoint, &byte_encoder_[static_cast<size_t>(byte)]);
        byte_decoder_[codepoint] = static_cast<uint8_t>(byte);
    }
    return 0;
}

std::string QwenTokenizer::EncodeBytes(const std::string& text) const {
    std::string encoded;
    for (const unsigned char byte : text) {
        encoded += byte_encoder_[byte];
    }
    return encoded;
}

std::string QwenTokenizer::DecodeBytes(const std::string& text) const {
    std::string decoded;
    for (const auto& unit : SplitUtf8(text)) {
        const auto it = byte_decoder_.find(unit.codepoint);
        if (it == byte_decoder_.end()) {
            decoded.append(text, unit.begin, unit.end - unit.begin);
        } else {
            decoded.push_back(static_cast<char>(it->second));
        }
    }
    return decoded;
}

std::vector<std::string> QwenTokenizer::Pretokenize(const std::string& text) const {
#ifdef FEATHER_WITH_ICU
    if (!split_pattern_.empty()) {
        UErrorCode error = U_ZERO_ERROR;
        const auto input = icu::UnicodeString::fromUTF8(text);
        icu::RegexMatcher matcher(icu::UnicodeString::fromUTF8(split_pattern_), input, 0, error);
        std::vector<std::string> pieces;
        while (U_SUCCESS(error) && matcher.find(error)) {
            std::string piece;
            input.tempSubStringBetween(matcher.start(error), matcher.end(error)).toUTF8String(piece);
            if (U_FAILURE(error)) return {};
            pieces.push_back(std::move(piece));
        }
        return U_SUCCESS(error) ? pieces : std::vector<std::string>{};
    }
#endif
    const auto units = SplitUtf8(text);
    std::vector<std::string> pieces;
    // Qwen3's ordered alternatives: contractions, optionally prefixed letters,
    // one number, punctuation with trailing newlines, then whitespace.
    for (size_t index = 0; index < units.size();) {
        const size_t begin = index;
        size_t contraction = 0;
        if (units[index].codepoint == '\'') {
            for (const std::string_view suffix : {"s", "t", "re", "ve", "m", "ll", "d"}) {
                if (index + suffix.size() >= units.size()) continue;
                bool matches = true;
                for (size_t i = 0; i < suffix.size(); ++i) {
                    auto c = units[index + 1 + i].codepoint;
                    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
                    matches &= c == static_cast<unsigned char>(suffix[i]);
                }
                if (matches) {
                    contraction = suffix.size() + 1;
                    break;
                }
            }
        }
        const auto c = units[index].codepoint;
        const bool prefix = !IsNewline(c) && !IsLetter(c) && !IsDigit(c) && index + 1 < units.size() &&
                            IsLetter(units[index + 1].codepoint);
        if (contraction) {
            index += contraction;
        } else if (IsLetter(c) || prefix) {
            if (prefix) ++index;
            while (index < units.size() && IsLetter(units[index].codepoint)) ++index;
        } else if (IsDigit(c)) {
            ++index;
        } else if (!IsWhitespace(c) ||
                   (c == ' ' && index + 1 < units.size() && !IsWhitespace(units[index + 1].codepoint) &&
                    !IsLetter(units[index + 1].codepoint) && !IsDigit(units[index + 1].codepoint))) {
            if (c == ' ') ++index;
            while (index < units.size() && !IsWhitespace(units[index].codepoint) && !IsLetter(units[index].codepoint) &&
                   !IsDigit(units[index].codepoint))
                ++index;
            while (index < units.size() && IsNewline(units[index].codepoint)) ++index;
        } else {
            size_t end = index, after_newline = index;
            while (end < units.size() && IsWhitespace(units[end].codepoint)) {
                if (IsNewline(units[end].codepoint)) after_newline = end + 1;
                ++end;
            }
            // \s*[\r\n]+ stops at the last newline. Otherwise \s+(?!\S)
            // leaves one whitespace before text for the prefixed-letter branch.
            index = after_newline > begin ? after_newline : end == units.size() || end == begin + 1 ? end : end - 1;
        }
        pieces.push_back(text.substr(units[begin].begin, units[index - 1].end - units[begin].begin));
    }
    return pieces;
}

std::vector<std::string> QwenTokenizer::ApplyBpe(const std::string& text) const {
    std::vector<std::string> symbols;
    const std::string encoded = EncodeBytes(text);
    for (const auto& unit : SplitUtf8(encoded)) {
        symbols.push_back(encoded.substr(unit.begin, unit.end - unit.begin));
    }
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = symbols.size();
        for (size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto it = merge_ranks_.find(PairKey(symbols[index], symbols[index + 1]));
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = index;
            }
        }
        if (best_index == symbols.size()) {
            break;
        }
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_index + 1));
    }
    return symbols;
}

std::vector<int64_t> QwenTokenizer::Encode(const std::string& text) const {
    std::vector<int64_t> ids;
    for (size_t begin = 0; begin < text.size();) {
        size_t special_position = text.size();
        std::string matched_special;
        for (const auto& special : special_tokens_by_length_) {
            const size_t position = text.find(special, begin);
            if (position < special_position) {
                special_position = position;
                matched_special = special;
            }
        }
        std::string regular = text.substr(begin, special_position - begin);
#ifdef FEATHER_WITH_ICU
        if (normalize_nfc_) {
            std::string normalized;
            if (!NormalizeNfc(regular, &normalized)) return {};
            regular = std::move(normalized);
        }
#endif
        for (const auto& piece : Pretokenize(regular)) {
            for (const auto& symbol : ApplyBpe(piece)) {
                const auto it = token_to_id_.find(symbol);
                if (it != token_to_id_.end()) {
                    ids.push_back(it->second);
                }
            }
        }
        if (matched_special.empty()) {
            break;
        }
        ids.push_back(special_token_to_id_.at(matched_special));
        begin = special_position + matched_special.size();
    }
    return ids;
}

std::string QwenTokenizer::Decode(const std::vector<int64_t>& token_ids) const {
    std::string decoded;
    for (const auto id : token_ids) {
        const auto special = special_id_to_token_.find(id);
        if (special != special_id_to_token_.end()) {
            decoded += special->second;
            continue;
        }
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() || id_to_token_[static_cast<size_t>(id)].empty()) {
            continue;
        }
        decoded += DecodeBytes(id_to_token_[static_cast<size_t>(id)]);
    }
    return decoded;
}

int64_t QwenTokenizer::TokenId(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

int64_t QwenTokenizer::EndOfTextTokenId() const { return TokenId("<|endoftext|>"); }

int64_t QwenTokenizer::ImEndTokenId() const { return TokenId("<|im_end|>"); }

}  // namespace demo
}  // namespace feather
