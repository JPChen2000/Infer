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

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "demo/qwen_chat_template.h"

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
    explicit ScopedWorkingDirectory(const std::filesystem::path& directory)
        : original_(std::filesystem::current_path()) {
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

#ifndef FEATHER_WITH_ICU
    GTEST_SKIP() << "Real Qwen tokenizer assets require ICU Unicode support";
#endif
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

#ifndef FEATHER_WITH_ICU
    GTEST_SKIP() << "Real Qwen tokenizer assets require ICU Unicode support";
#endif
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

#ifndef FEATHER_WITH_ICU
    GTEST_SKIP() << "Real Qwen tokenizer assets require ICU Unicode support";
#endif
    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(kTokenizerDir.string()), 0) << tokenizer.LastError();

    const auto prompt =
        feather::demo::RenderQwenChat({{"system", "You are a helpful assistant."}, {"user", "介绍一下你自己"}}, true);
    EXPECT_EQ(tokenizer.Encode(prompt),
              (std::vector<int64_t>{248045, 8678,  198,    2523,   513, 264,    10631,  17313,  13,
                                    248046, 198,   248045, 846,    198, 113552, 111522, 248046, 198,
                                    248045, 74455, 198,    248068, 271, 248069, 271}));
}

TEST(qwen_tokenizer_test, MatchesQwen3ContractionsPrefixesAndWhitespace) {
    TokenizerAssetFixture fixture;
    ASSERT_TRUE(fixture.CreateFile("vocab.json"));
    // Subset of the Qwen/Qwen3-4B vocabulary and merge ancestry for these cases.
    // Expected IDs were obtained independently with the official tokenizer.json.
    // All assets are embedded so this regression needs no checkpoint or Python.
    std::ofstream(fixture.directory() / "vocab.json")
        << R"qwen({"\"":1,"1":16,"Ô":144,"č":201,"is":285,".":13,"ë":167,"wo":1126,"2":17,"ot":354,"?":30,"sm":3563,"ğ":219,"É":133,"ä¸ĸ":99244,"Ã":127,"Ĉ":196,")":8,"²":110,"x":87,"ù":181,"Ę":212,"Ĩ":228,"q":80,"d":67,"ñ":173,"Ą":192,"ĸ":244,"·":115,"Ħ":226,"čĊ":319,"ile":457,"E":36,"ä¸Ń":15946,"red":1151,"ā":189,"ï":171,"re":265,"M":44,"Y":56,"ļ":248,"ì":168,"ĵ":241,"Ġsmiling":36063,"¢":95,"ê":166,":":25,"Ğ":218,"X":55,"ę":213,"§":100,"ġ":221,"©":102,"]":60,"Č":200,"Ď":202,"Ö":146,"aired":24274,"Ĕ":208,"#":2,"'t":944,"(":7,"Ġwom":2358,"ired":2690,"ĝ":217,"¾":122,"ãĢ":1277,"ĩ":229,"m":76,"s":82,"â":158,"á":157,"ģ":223,"Ġh":305,"f":69,"k":74,"I":40,"ě":215,"Ġs":274,"ĠTHE":3168,"b":65,"Ú":150,"t":83,"Đ":204,"°":108,"om":316,"¹":117,"in":258,"ľ":250,"ðŁ":9284,"=":28,"F":37,"ð":172,"*":9,"Æ":130,"Ė":210,"Å":129,"Ë":135,"À":124,"ª":103,"¤":97,"ĻĤ":18535,"¼":120,"Ø":148,"5":20,"Ġb":293,"two":19789,"ä¸":3490,"r":81,"¬":105,"û":183,"K":42,"'M":27603,"Ċ":198,"ĭ":233,"j":73,"İ":236,"å¥½":52801,"H":39,"-":12,"Ě":214,"Ĵ":240,"ŀ":252,"åı¯çĪ±":102783,"çķĮ":97120,"le":273,"×":147,"ć":195,"ó":175,">":29,"Ľ":249,"ĠW":467,"&":5,"9":24,"A":32,"ĠĠĠ":262,"ĠT":350,"Ġo":297,"O":46,"U":52,"Ġa":264,"ç":163,"Õ":145,"Ļ":247,"a":64,"þ":186,"y":88,"çĮ«":100472,",":11,"Ġis":374,"7":22,"ĠTHEY":62493,"®":106,"Ġw":289,"P":47,"4":19,"ire":554,";":26,"ď":203,"¶":114,"į":235,"Ē":206,"@":31,"ċ":199,"'s":594,"ĉword":66298,"Ġp":281,"Ł":253,"åı":5758,"ĕ":209,"n":77,"¨":101,"¡":94,"<":27,"ĠĉĊ":17476,"Q":48,"ķ":243,"ðŁĲ":123808,"Ġph":1319,"R":49,"or":269,"3":18,"Þ":154,"ä":160,"ý":185,"an":276,"g":70,"Ġof":315,"đ":205,"ò":174,"Ģ":222,"ô":176,"È":132,"ė":211,"çĮ":99177,"æĸ":7588,"î":170,"¥":98,"Ĭ":232,"çķ":39098,"ĠWE":19677,"å¥":27095,"±":109,"Ġcat":8251,"Á":125,"Ā":188,"{":90,"Ü":152,"N":45,"÷":179,"ä½":8519,"HE":1799,"VE":4491,"ĺ":246,"º":118,"¯":107,"^":61,"ø":180,"ł":254,"ĳ":239,"J":41,"ĠI":358,"ãĢĤ":1773,"es":288,"[":58,"í":169,"T":51,"D":35,"Ă":190,"ĉ":197,"ĉĊ":1572,"+":10,"ord":539,"ß":155,"Į":234,"S":50,"ÿ":187,"ĠĠ":256,"Ä":128,"çĪ±":99242,"ï¼ģ":6313,"µ":113,"Ý":153,"Ć":194,"W":54,"Î":138,"}":92,"Ê":134,"Ò":142,"Ù":149,"LL":4086,"L":43,"åı¯":30440,"Ī":230,"h":71,"ä½łå¥½":108386,"C":34,"¸":116,"Ç":131,"ú":182,"ä½ł":56568,"Ġsm":1525,"$":3,"è":164,"Ġsp":978,"æĸĩ":16744,"Ġhat":8896,"word":1158,"Ń":255,"ï¼Į":3837,"oto":2072,"Ġ":220,"ðŁĻĤ":145080,"'":6,"at":266,"u":84,"B":33,"ö":178,"ï¼":1540,"/":14,"`":63,"ē":207,"6":21,"!":0,"Ŀ":251,"'D":27705,"0":15,"½":121,"Ĳ":238,"çĪ":75107,"i":72,"¦":99,"Ġphoto":6548,"ī":231,"Ġc":272,"ħ":227,"ă":191,"Ĺ":245,"G":38,"ã":159,"o":78,"p":79,"ĥ":225,"_":62,"Í":137,"%":4,"ac":580,"Ġwoman":5220,"ı":237,"e":68,"£":96,"V":53,"ing":287,"Ï":139,"w":86,"~":93,"»":119,"Ġspaces":12621,"ðŁĲĪ":146833,"Z":57,"³":111,"Ñ":141,"Ġisn":4436,"ą":193,"«":104,"Û":151,"Â":126,"il":321,"'RE":94153,"ü":184,"Ĝ":216,"é":165,"à":156,"z":89,"´":112,"ä¸ĸçķĮ":99489,"Ó":143,"|":91,"Ĥ":224,"Ķ":242,"aces":2434,"l":75,"å":161,"ä¸Ńæĸĩ":104811,"iling":7979,"õ":177,"RE":787,"Ì":136,"c":66,"Ð":140,"æ":162,"8":23,"¿":123,"\\":59,"-h":2832,"v":85,"-haired":67379})qwen";
    std::ofstream(fixture.directory() / "merges.txt") << R"qwen(#version: 0.2
Ġ Ġ
i n
ĠĠ Ġ
Ġ a
r e
a t
o r
Ġ c
l e
Ġ s
a n
Ġ p
i s
in g
e s
Ġ w
Ġ b
Ġ o
Ġ h
Ġo f
o m
č Ċ
i l
Ġ T
o t
Ġ I
Ġ is
i le
Ġ W
or d
i re
a c
' s
R E
' t
Ġs p
w o
re d
w ord
ã Ģ
Ġp h
Ġs m
ï ¼
ĉ Ċ
ãĢ Ĥ
H E
ot o
Ġw om
ac es
ire d
- h
ĠT HE
ä ¸
s m
ï¼ Į
L L
Ġis n
V E
Ġwom an
å ı
ï¼ ģ
Ġph oto
æ ĸ
il ing
Ġc at
ä ½
Ġh at
ð Ł
Ġsp aces
ä¸ Ń
æĸ ĩ
Ġ ĉĊ
Ļ Ĥ
ĠW E
t wo
a ired
å ¥
' M
' D
åı ¯
Ġsm iling
ç ķ
å¥ ½
ä½ ł
ĠTHE Y
ĉ word
-h aired
ç Ī
' RE
çķ Į
ç Į
çĪ ±
ä¸ ĸ
ä¸ĸ çķĮ
çĮ «
åı¯ çĪ±
ä¸Ń æĸĩ
ä½ł å¥½
ðŁ Ĳ
ðŁ ĻĤ
ðŁĲ Ī
)qwen";
    std::ofstream(fixture.directory() / "tokenizer.json")
        << R"({"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151644,"content":"<|im_start|>"},{"id":151645,"content":"<|im_end|>"}]})";
    feather::demo::QwenTokenizer tokenizer;
    ASSERT_EQ(tokenizer.Load(fixture.directory().string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode("A cat isn't smiling"), (std::vector<int64_t>{32, 8251, 4436, 944, 36063}));
    EXPECT_EQ(tokenizer.Encode("A cat's hat"), (std::vector<int64_t>{32, 8251, 594, 8896}));
    EXPECT_EQ(tokenizer.Encode("I'M I'LL WE'RE THEY'VE I'D"),
              (std::vector<int64_t>{40, 27603, 358, 6, 4086, 19677, 94153, 62493, 6, 4491, 358, 27705}));
    EXPECT_EQ(tokenizer.Encode("red-haired woman"), (std::vector<int64_t>{1151, 67379, 5220}));
    EXPECT_EQ(tokenizer.Encode("two  spaces"), (std::vector<int64_t>{19789, 220, 12621}));
    EXPECT_EQ(tokenizer.Encode("a\n b"), (std::vector<int64_t>{64, 198, 293}));
    EXPECT_EQ(tokenizer.Encode("word\tword"), (std::vector<int64_t>{1158, 66298}));
    EXPECT_EQ(tokenizer.Encode("a \t\n\r\n b"), (std::vector<int64_t>{64, 17476, 319, 293}));
    EXPECT_EQ(tokenizer.Encode("a   "), (std::vector<int64_t>{64, 262}));
    EXPECT_EQ(tokenizer.Encode("中文，世界！你好。"), (std::vector<int64_t>{104811, 3837, 99489, 6313, 108386, 1773}));
    EXPECT_EQ(tokenizer.Encode("猫🐈可爱"), (std::vector<int64_t>{100472, 146833, 102783}));
    EXPECT_EQ(tokenizer.Encode("🙂smile"), (std::vector<int64_t>{145080, 3563, 457}));
    EXPECT_EQ(tokenizer.Encode("A photo of a smiling woman."),
              (std::vector<int64_t>{32, 6548, 315, 264, 36063, 5220, 13}));
}

namespace {
bool WriteUnicodeTokenizer(const TokenizerAssetFixture& fixture, const std::string& normalizer,
                           const std::string& pattern = "") {
    if (!fixture.CreateFile("vocab.json")) return false;
    std::ofstream(fixture.directory() / "vocab.json")
        << R"({"a":0,"b":1,"Ì":2,"¸":3,"Ì¸":4,"Ì¸b":5,"aÌ¸b":6,"Ã":7,"©":8,"Ã©":9,"e":10,"ģ":11})";
    std::ofstream(fixture.directory() / "merges.txt") << "#version: 0.2\nÌ ¸\nÌ¸ b\na Ì¸b\nÃ ©\n";
    std::ofstream config(fixture.directory() / "tokenizer.json");
    config << R"({"added_tokens":[],"normalizer":)" << normalizer;
    if (!pattern.empty())
        config << R"(,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":")"
               << pattern << R"("},"behavior":"Isolated","invert":false},{"type":"ByteLevel","use_regex":false}]})";
    config << '}';
    return config.good();
}
}  // namespace

TEST(qwen_tokenizer_test, RejectsUnsupportedNormalizer) {
    TokenizerAssetFixture fixture;
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, R"({"type":"NFD"})"));
    feather::demo::QwenTokenizer tokenizer;
    EXPECT_NE(tokenizer.Load(fixture.directory().string()), 0);
    EXPECT_NE(tokenizer.LastError().find("normalizer"), std::string::npos);
}

TEST(qwen_tokenizer_test, HonorsConfiguredNfcNormalization) {
    TokenizerAssetFixture fixture;
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, R"({"type":"NFC"})"));
    feather::demo::QwenTokenizer tokenizer;
#ifdef FEATHER_WITH_ICU
    ASSERT_EQ(tokenizer.Load(fixture.directory().string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode("e\u0301"), (std::vector<int64_t>{9}));
    EXPECT_EQ(tokenizer.Encode("é"), (std::vector<int64_t>{9}));
    // A subsequent non-normalizing tokenizer must not retain the previous mode.
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, "null"));
    ASSERT_EQ(tokenizer.Load(fixture.directory().string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode("e\u0301"), (std::vector<int64_t>{10, 2, 11}));
#else
    EXPECT_NE(tokenizer.Load(fixture.directory().string()), 0);
    EXPECT_NE(tokenizer.LastError().find("ICU"), std::string::npos);
#endif
}

TEST(qwen_tokenizer_test, PreservesConfiguredQwen3AndQwen35MarkBoundaries) {
    TokenizerAssetFixture fixture;
    feather::demo::QwenTokenizer tokenizer;
    // The two real checkpoint families differ in whether a letter run includes
    // combining marks. This accent does not compose under NFC.
    const std::string qwen3 =
        R"([^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
    const std::string qwen35 =
        R"([^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, R"({"type":"NFC"})", qwen3));
#ifdef FEATHER_WITH_ICU
    ASSERT_EQ(tokenizer.Load(fixture.directory().string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode("a\u0338b"), (std::vector<int64_t>{0, 5}));
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, R"({"type":"NFC"})", qwen35));
    ASSERT_EQ(tokenizer.Load(fixture.directory().string()), 0) << tokenizer.LastError();
    EXPECT_EQ(tokenizer.Encode("a\u0338b"), (std::vector<int64_t>{6}));
#else
    EXPECT_NE(tokenizer.Load(fixture.directory().string()), 0);
    EXPECT_NE(tokenizer.LastError().find("ICU"), std::string::npos);
    // Regex configuration alone also needs ICU, even without normalization.
    ASSERT_TRUE(WriteUnicodeTokenizer(fixture, "null", qwen35));
    EXPECT_NE(tokenizer.Load(fixture.directory().string()), 0);
    EXPECT_NE(tokenizer.LastError().find("ICU"), std::string::npos);
#endif
}
