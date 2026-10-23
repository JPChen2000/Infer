// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "demo/flux2_runner.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>

#include "model/model_io.h"
namespace {
using namespace feather;
using namespace feather::demo;
model::ValueDesc Value(std::string name, std::vector<int64_t> dims, DataType dtype = DataType::FP32) {
    model::ValueDesc v;
    v.tensor.name = std::move(name);
    v.tensor.dims = std::move(dims);
    v.tensor.data_type = dtype;
    return v;
}
struct Assets {
    std::filesystem::path dir;
    model::ModelDesc text, denoise, vae;
    Assets() {
        static std::atomic<unsigned> serial{0};
        dir = std::filesystem::temp_directory_path() / ("feather-flux2-" + std::to_string(++serial));
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "vocab.json") << R"({"a":0,"b":1})";
        std::ofstream(dir / "merges.txt") << "#version: 0.2\na b\n";
        std::ofstream(dir / "tokenizer.json")
            << R"({"added_tokens":[{"id":2,"content":"<|endoftext|>"},{"id":3,"content":"<|im_start|>"},{"id":4,"content":"<|im_end|>"}]})";
        for (auto* m : {&text, &denoise, &vae}) {
            m->name = "tiny_flux2";
            m->version = 1;
            m->graph.name = "main";
            m->metadata = {{"flux2.schema", int64_t(1)},           {"flux2.variant", std::string("klein-4b")},
                           {"flux2.height", int64_t(16)},          {"flux2.width", int64_t(16)},
                           {"flux2.text_length", int64_t(8)},      {"flux2.context_dim", int64_t(1)},
                           {"flux2.latent_channels", int64_t(768)}};
        }
        text.metadata["flux2.component"] = std::string("text_encoder");
        text.graph.inputs = {"input_ids", "attention_mask"};
        text.graph.outputs = {"prompt_embeds"};
        text.graph.values = {Value("input_ids", {1, 8}, DataType::INT64), Value("attention_mask", {1, 1, 1, 8}),
                             Value("embedding", {1, 8, 1}), Value("prompt_embeds", {1, 8, 1})};
        text.graph.values[2].constant = true;
        model::NodeDesc node;
        node.name = "copy";
        node.op_type = "Identity";
        node.inputs = {"embedding"};
        node.outputs = {"prompt_embeds"};
        text.graph.nodes = {node};
        denoise.metadata["flux2.component"] = std::string("denoiser");
        denoise.graph.inputs = {"latents", "prompt_embeds", "timestep"};
        denoise.graph.outputs = {"velocity"};
        denoise.graph.values = {Value("latents", {1, 1, 768}), Value("prompt_embeds", {1, 8, 1}),
                                Value("timestep", {1}), Value("velocity", {1, 1, 768})};
        node.inputs = {"latents"};
        node.outputs = {"velocity"};
        denoise.graph.nodes = {node};
        vae.metadata["flux2.component"] = std::string("vae_decoder");
        vae.graph.inputs = {"latents"};
        vae.graph.outputs = {"image"};
        vae.graph.values = {Value("latents", {1, 1, 768}), Value("image", {1, 3, 16, 16})};
        node.op_type = "Reshape";
        node.inputs = {"latents"};
        node.outputs = {"image"};
        node.attributes["shape"] = std::vector<int64_t>{1, 3, 16, 16};
        vae.graph.nodes = {node};
    }
    ~Assets() {
        std::error_code e;
        std::filesystem::remove_all(dir, e);
    }
    bool Save(float embedding_value = .25f) {
        model::ModelWriter w;
        auto embedding = std::make_shared<Tensor>();
        embedding->Assign(std::vector<float>(8, embedding_value), {1, 8, 1});
        return w.Save((dir / "text_encoder.fth").string(), text, {{"embedding", embedding}}) &&
               w.Save((dir / "denoiser.fth").string(), denoise, {}) &&
               w.Save((dir / "vae_decoder.fth").string(), vae, {});
    }
};
TEST(flux2_runner_test, OfficialScheduleAndEndpoints) {
    auto t = Flux2Runner::Timesteps(4096, 4);
    ASSERT_EQ(t.size(), 5u);
    EXPECT_EQ(t.front(), 1.f);
    EXPECT_EQ(t.back(), 0.f);
    EXPECT_NEAR(t[2], .90814392f, 1e-6f);
    for (size_t i = 1; i < t.size(); ++i) EXPECT_LT(t[i], t[i - 1]);
    EXPECT_TRUE(Flux2Runner::Timesteps(1, 0).empty());
}
TEST(flux2_runner_test, ExecutesThreeStagesAndRepeatsDeterministically) {
    Assets a;
    ASSERT_TRUE(a.Save());
    Flux2Runner r;
    ASSERT_TRUE(r.Load(a.dir.string()).ok()) << r.LastError();
    ImageData x, y;
    ASSERT_TRUE(r.Generate("ab", &x).ok()) << r.LastError();
    ASSERT_TRUE(r.Generate("a", &y).ok()) << r.LastError();
    EXPECT_EQ(x.pixels, y.pixels);
    EXPECT_EQ(x.width, 16);
    EXPECT_EQ(x.channels, 3);
    EXPECT_EQ(x.pixels.size(), 768u);
    ASSERT_TRUE(r.GenerateFromEmbeddings(std::vector<float>(8, .5f), &y).ok()) << r.LastError();
    EXPECT_EQ(x.pixels, y.pixels);
    bool varied = false;
    for (auto p : x.pixels) varied |= p != x.pixels[0];
    EXPECT_TRUE(varied);
}
TEST(flux2_runner_test, RejectsContractsAndInvalidOptions) {
    Assets a;
    a.denoise.metadata["flux2.schema"] = int64_t(2);
    ASSERT_TRUE(a.Save());
    Flux2Runner r;
    EXPECT_FALSE(r.Load(a.dir.string()).ok());
    a.denoise.metadata["flux2.schema"] = int64_t(1);
    a.vae.graph.values[1].tensor.dims = {1, 3, 16, 15};
    ASSERT_TRUE(a.Save());
    EXPECT_FALSE(r.Load(a.dir.string()).ok());
    Flux2Options o;
    o.steps = 0;
    EXPECT_FALSE(r.Load(a.dir.string(), o).ok());
}
TEST(flux2_runner_test, RejectsWrongOptionalMetadataTypes) {
    Assets a;
    Flux2Runner r;
    a.text.metadata["flux2.padding_side"] = int64_t(1);
    ASSERT_TRUE(a.Save());
    EXPECT_FALSE(r.Load(a.dir.string()).ok());
    a.text.metadata["flux2.padding_side"] = std::string("right");
    a.text.metadata["flux2.pad_token_id"] = std::string("2");
    ASSERT_TRUE(a.Save());
    EXPECT_FALSE(r.Load(a.dir.string()).ok());
}
TEST(flux2_runner_test, RejectsNonfiniteStageOutput) {
    Assets a;
    ASSERT_TRUE(a.Save(std::numeric_limits<float>::quiet_NaN()));
    Flux2Runner r;
    ASSERT_TRUE(r.Load(a.dir.string()).ok());
    ImageData image;
    image.width = 99;
    EXPECT_FALSE(r.Generate("ab", &image).ok());
    EXPECT_NE(r.LastError().find("nonfinite"), std::string::npos);
    EXPECT_EQ(image.width, 99);
}
TEST(flux2_runner_test, FailurePreservesResultAndRejectsNonfiniteEmbeddings) {
    Assets a;
    ASSERT_TRUE(a.Save());
    Flux2Runner r;
    ASSERT_TRUE(r.Load(a.dir.string()).ok()) << r.LastError();
    ImageData image;
    image.width = 123;
    image.pixels = {4, 5};
    EXPECT_FALSE(r.GenerateFromEmbeddings({}, &image).ok());
    auto emb = std::vector<float>(8, 0);
    emb[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(r.GenerateFromEmbeddings(emb, &image).ok());
    EXPECT_EQ(image.width, 123);
    EXPECT_EQ(image.pixels, (std::vector<uint8_t>{4, 5}));
    std::filesystem::remove(a.dir / "vae_decoder.fth");
    EXPECT_FALSE(r.GenerateFromEmbeddings(std::vector<float>(8, 0), &image).ok());
    EXPECT_EQ(image.width, 123);
}
}  // namespace
