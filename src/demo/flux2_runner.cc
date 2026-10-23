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

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <set>

#include "core/execution_plan.h"
#include "model/model_io.h"
#include "util/bf16.h"
#include "util/fp16.h"
namespace feather {
namespace demo {
namespace {
Status Error(const std::string& s) { return Status::Error(StatusCode::kInvalidArgument, s); }
bool Floating(DataType d) { return d == DataType::FP32 || d == DataType::FP16 || d == DataType::BF16; }
const model::TensorDesc* Desc(const model::ModelDesc& m, const std::string& n) {
    for (const auto& v : m.graph.values)
        if (v.tensor.name == n) return &v.tensor;
    return nullptr;
}
template <class T>
const T* Meta(const model::ModelDesc& m, const std::string& n) {
    auto i = m.metadata.find("flux2." + n);
    return i == m.metadata.end() ? nullptr : std::get_if<T>(&i->second);
}
std::shared_ptr<Tensor> Make(const model::TensorDesc& d, const std::vector<float>& v) {
    auto t = std::make_shared<Tensor>(d.dims);
    t->set_layout(d.layout);
    if (d.data_type == DataType::FP32)
        std::copy(v.begin(), v.end(), t->mutable_data<float>());
    else {
        auto p = t->mutable_data<uint16_t>();
        for (size_t i = 0; i < v.size(); ++i)
            p[i] = d.data_type == DataType::BF16 ? FloatToBFloat16(v[i]) : FloatToHalf(v[i]);
    }
    t->set_data_type(d.data_type);
    return t;
}
Status Read(ExecutionSession& s, const std::string& name, std::vector<float>* v) {
    std::shared_ptr<Tensor> t;
    auto st = s.ReadOutput(name, &t);
    if (!st.ok()) return st;
    v->resize(t->numel());
    for (size_t i = 0; i < v->size(); ++i) {
        float x = t->data_type() == DataType::FP32   ? t->data<float>()[i]
                  : t->data_type() == DataType::BF16 ? BFloat16ToFloat(t->data<uint16_t>()[i])
                                                     : HalfToFloat(t->data<uint16_t>()[i]);
        if (!std::isfinite(x)) return Error("nonfinite output: " + name);
        (*v)[i] = x;
    }
    return Status::Ok();
}
struct Stage {
    model::ModelLoader loader;
    std::shared_ptr<const CompiledGraph> compiled;
    std::unique_ptr<ExecutionSession> session;
    Status Open(const std::string& path, Flux2Backend backend) {
        if (!loader.Load(path)) return Error("cannot load component: " + path);
        StaticGraph g;
        DeviceType device = backend == Flux2Backend::kCuda     ? DeviceType::CUDA
                            : backend == Flux2Backend::kCommon ? DeviceType::COMMON
                            : backend == Flux2Backend::kX86    ? DeviceType::X86
                                                               : GetHostRuntimeDevice();
        g.SetKernelDevice(device);
        if (device == DeviceType::CUDA) g.SetFallbackPolicy(KernelFallbackPolicy::kRequireRequested);
        if (g.SetModel(loader.model()) != 0) return Error("cannot set component graph");
        for (const auto& v : loader.model().graph.values)
            if (v.constant) {
                auto t = loader.CreateWeightTensor(v.tensor.name);
                if (!t || g.SetTensor(v.tensor.name, t) != 0) return Error("cannot load weight: " + v.tensor.name);
            }
        for (const auto& name : loader.model().graph.inputs) {
            const auto* d = Desc(loader.model(), name);
            if (!d) return Error("missing input descriptor: " + name);
            std::shared_ptr<Tensor> t;
            if (d->data_type == DataType::INT64) {
                t = std::make_shared<Tensor>();
                size_t count = 0;
                if (!CheckedElementCount(d->dims, &count)) return Error("invalid input shape");
                t->Assign(std::vector<int64_t>(count, 0), d->dims);
                t->set_layout(d->layout);
            } else {
                size_t count = 0;
                if (!CheckedElementCount(d->dims, &count)) return Error("invalid input shape");
                t = Make(*d, std::vector<float>(count, 0));
            }
            if (g.SetTensor(name, t) != 0) return Error("cannot attach input: " + name);
        }
        auto st = g.BuildStatus();
        if (!st.ok()) return st;
        st = CompiledGraph::Compile(g, &compiled);
        if (!st.ok()) return st;
        return compiled->CreateSession(&session);
    }
    Status Bind(const std::string& n, const std::vector<float>& v) {
        return session->BindInput(n, Make(*Desc(loader.model(), n), v));
    }
};
}  // namespace
bool ParseFlux2Backend(const std::string& s, Flux2Backend* b) {
    if (!b) return false;
    if (s == "common")
        *b = Flux2Backend::kCommon;
    else if (s == "host")
        *b = Flux2Backend::kHost;
    else if (s == "x86")
        *b = Flux2Backend::kX86;
    else if (s == "cuda")
        *b = Flux2Backend::kCuda;
    else
        return false;
    return true;
}
Status Flux2Runner::Finish(Status s) {
    last_error_ = s.message();
    return s;
}
Status Flux2Runner::Load(const std::string& dir, const Flux2Options& o) {
    loaded_ = false;
    last_error_.clear();
    if (o.steps <= 0 || o.steps > 10000) return Finish(Error("steps must be in [1,10000]"));
#ifndef FEATHER_WITH_CUDA
    if (o.backend == Flux2Backend::kCuda) return Finish(Error("CUDA requested but this build has no CUDA support"));
#endif
    const char* names[] = {"text_encoder", "denoiser", "vae_decoder"};
    model::ModelDesc* models[] = {&text_, &denoiser_, &decoder_};
    for (int i = 0; i < 3; ++i) {
        model::ModelLoader loader;
        if (!loader.Load(dir + "/" + names[i] + ".fth"))
            return Finish(Error("cannot load " + dir + "/" + names[i] + ".fth"));
        *models[i] = loader.model();
        auto& m = *models[i];
        auto schema = Meta<int64_t>(m, "schema");
        auto component = Meta<std::string>(m, "component");
        auto variant = Meta<std::string>(m, "variant");
        if (!schema || *schema != 1 || !component || *component != names[i] || !variant || *variant != "klein-4b")
            return Finish(Error("invalid FLUX.2 schema/component/variant: " + std::string(names[i])));
        for (const auto* key : {"height", "width", "text_length", "context_dim", "latent_channels"}) {
            auto value = Meta<int64_t>(m, key);
            auto base = Meta<int64_t>(text_, key);
            if (!value || *value <= 0 || !base || *value != *base)
                return Finish(Error("inconsistent FLUX.2 metadata: " + std::string(key)));
        }
    }
    height_ = *Meta<int64_t>(text_, "height");
    width_ = *Meta<int64_t>(text_, "width");
    length_ = *Meta<int64_t>(text_, "text_length");
    context_ = *Meta<int64_t>(text_, "context_dim");
    channels_ = *Meta<int64_t>(text_, "latent_channels");
    if (height_ % 16 || width_ % 16 || height_ > 32768 || width_ > 32768 || length_ > 1048576 || context_ > 1048576 ||
        channels_ > 1048576)
        return Finish(Error("invalid fixed dimensions"));
    const std::vector<int64_t> lat = {1, (height_ / 16) * (width_ / 16), channels_}, emb = {1, length_, context_};
    auto check = [&](const model::ModelDesc& m, const std::string& n, const std::vector<int64_t>& dims,
                     DataType dtype) {
        auto d = Desc(m, n);
        size_t count = 0;
        return d && d->dims == dims && CheckedElementCount(dims, &count) &&
               count <= static_cast<size_t>(std::numeric_limits<int64_t>::max() / 8) &&
               (dtype == DataType::UNKNOWN ? Floating(d->data_type) : d->data_type == dtype);
    };
    auto io = [](const model::ModelDesc& m, std::set<std::string> in, std::set<std::string> out) {
        return std::set<std::string>(m.graph.inputs.begin(), m.graph.inputs.end()) == in &&
               std::set<std::string>(m.graph.outputs.begin(), m.graph.outputs.end()) == out;
    };
    if (!io(text_, {"input_ids", "attention_mask"}, {"prompt_embeds"}) ||
        !io(denoiser_, {"latents", "prompt_embeds", "timestep"}, {"velocity"}) ||
        !io(decoder_, {"latents"}, {"image"}) || !check(text_, "input_ids", {1, length_}, DataType::INT64) ||
        !check(text_, "attention_mask", {1, 1, 1, length_}, DataType::FP32) ||
        !check(text_, "prompt_embeds", emb, DataType::UNKNOWN) ||
        !check(denoiser_, "prompt_embeds", emb, DataType::UNKNOWN) ||
        !check(denoiser_, "latents", lat, DataType::UNKNOWN) || !check(denoiser_, "velocity", lat, DataType::UNKNOWN) ||
        !check(denoiser_, "timestep", {1}, DataType::FP32) || !check(decoder_, "latents", lat, DataType::UNKNOWN) ||
        !check(decoder_, "image", {1, 3, height_, width_}, DataType::FP32))
        return Finish(Error("invalid component input/output contract"));
    if (Desc(denoiser_, "velocity")->data_type != Desc(denoiser_, "latents")->data_type)
        return Finish(Error("velocity dtype differs from latents"));
    tokenizer_ready_ = tokenizer_.Load(dir) == 0;
    if (tokenizer_ready_ && (tokenizer_.TokenId("<|im_start|>") < 0 || tokenizer_.ImEndTokenId() < 0))
        return Finish(Error("Qwen3 special tokens missing"));
    auto pad = Meta<int64_t>(text_, "pad_token_id");
    if (!pad && text_.metadata.count("flux2.pad_token_id")) return Finish(Error("pad_token_id must be an int64"));
    pad_ = pad ? *pad : tokenizer_.EndOfTextTokenId();
    if (tokenizer_ready_ && pad_ < 0) return Finish(Error("missing pad token"));
    auto side = Meta<std::string>(text_, "padding_side");
    if (!side && text_.metadata.count("flux2.padding_side")) return Finish(Error("padding_side must be a string"));
    if (side && *side != "left" && *side != "right") return Finish(Error("invalid padding_side"));
    left_padding_ = side && *side == "left";
    directory_ = dir;
    options_ = o;
    loaded_ = true;
    return Finish(Status::Ok());
}
std::vector<float> Flux2Runner::Timesteps(int64_t seq, int steps) {
    if (seq <= 0 || steps <= 0 || steps > 10000) return {};
    double m200 = .00016927 * seq + .45666666, m10 = 8.73809524e-05 * seq + 1.89833333;
    double a = (m200 - m10) / 190., mu = seq > 4300 ? m200 : a * steps + m200 - 200 * a, shift = std::exp(mu);
    std::vector<float> ts(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        double t = 1. - double(i) / steps;
        ts[i] = i == 0 ? 1.f : i == steps ? 0.f : static_cast<float>(shift * t / (shift * t + 1. - t));
    }
    return ts;
}
Status Flux2Runner::Generate(const std::string& prompt, ImageData* image) {
    if (!loaded_ || !image) return Finish(Error("runner not loaded or null image"));
    if (!tokenizer_ready_) return Finish(Error("tokenizer unavailable: " + tokenizer_.LastError()));
    auto ids =
        tokenizer_.Encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    if (ids.empty()) return Finish(Error("empty tokenized prompt"));
    if (ids.size() > static_cast<size_t>(length_)) ids.resize(length_);
    size_t used = ids.size();
    std::vector<int64_t> padded(length_, pad_);
    size_t start = left_padding_ ? length_ - used : 0;
    std::copy(ids.begin(), ids.end(), padded.begin() + start);
    std::vector<float> mask(length_, -std::numeric_limits<float>::infinity());
    std::fill(mask.begin() + start, mask.begin() + start + used, 0.f);
    std::vector<float> embeddings;
    {
        Stage stage;
        auto st = stage.Open(directory_ + "/text_encoder.fth", options_.backend);
        if (!st.ok()) return Finish(st);
        auto input = std::make_shared<Tensor>();
        input->Assign(padded, {1, length_});
        input->set_layout(Desc(text_, "input_ids")->layout);
        st = stage.session->BindInput("input_ids", input);
        if (st.ok()) st = stage.Bind("attention_mask", mask);
        if (st.ok()) st = stage.session->Run();
        if (st.ok()) st = Read(*stage.session, "prompt_embeds", &embeddings);
        if (!st.ok()) return Finish(st.WithContext("text encoder"));
    }
    return Finish(GenerateImpl(embeddings, image));
}
Status Flux2Runner::GenerateFromEmbeddings(const std::vector<float>& embeddings, ImageData* image) {
    return Finish(GenerateImpl(embeddings, image));
}
Status Flux2Runner::GenerateImpl(const std::vector<float>& emb, ImageData* image) {
    if (!loaded_ || !image || emb.size() != static_cast<size_t>(length_ * context_))
        return Error("not loaded, null image, or incorrect embedding size");
    for (float x : emb)
        if (!std::isfinite(x)) return Error("nonfinite embeddings");
    size_t seq = (height_ / 16) * (width_ / 16);
    std::vector<float> lat(seq * channels_);
    std::mt19937 rng(options_.seed);
    for (size_t i = 0; i < lat.size(); i += 2) {
        double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
        double r = std::sqrt(-2. * std::log(u)), theta = 6.2831853071795864769 * v;
        lat[i] = r * std::cos(theta);
        if (i + 1 < lat.size()) lat[i + 1] = r * std::sin(theta);
    }
    auto times = Timesteps(seq, options_.steps);
    {
        Stage stage;
        auto st = stage.Open(directory_ + "/denoiser.fth", options_.backend);
        if (!st.ok()) return st.WithContext("denoiser");
        st = stage.Bind("prompt_embeds", emb);
        if (!st.ok()) return st;
        for (int i = 0; i < options_.steps; ++i) {
            st = stage.Bind("latents", lat);
            if (st.ok()) st = stage.Bind("timestep", {times[i]});
            if (st.ok()) st = stage.session->Run();
            std::vector<float> velocity;
            if (st.ok()) st = Read(*stage.session, "velocity", &velocity);
            if (!st.ok()) return st.WithContext("denoiser");
            if (velocity.size() != lat.size()) return Error("invalid velocity size");
            for (size_t j = 0; j < lat.size(); ++j) {
                lat[j] += (times[i + 1] - times[i]) * velocity[j];
                if (!std::isfinite(lat[j])) return Error("nonfinite Euler latent");
            }
        }
    }
    std::vector<float> pixels;
    {
        Stage stage;
        auto st = stage.Open(directory_ + "/vae_decoder.fth", options_.backend);
        if (st.ok()) st = stage.Bind("latents", lat);
        if (st.ok()) st = stage.session->Run();
        if (st.ok()) st = Read(*stage.session, "image", &pixels);
        if (!st.ok()) return st.WithContext("VAE decoder");
    }
    size_t plane = height_ * width_;
    if (pixels.size() != 3 * plane) return Error("invalid image size");
    ImageData result;
    result.height = height_;
    result.width = width_;
    result.channels = 3;
    result.pixels.resize(3 * plane);
    for (size_t p = 0; p < plane; ++p)
        for (size_t c = 0; c < 3; ++c)
            result.pixels[3 * p + c] = static_cast<uint8_t>(
                std::round(std::max(0.f, std::min(1.f, .5f * pixels[c * plane + p] + .5f)) * 255.f));
    *image = std::move(result);
    return Status::Ok();
}
}  // namespace demo
}  // namespace feather
