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

#include <gtest/gtest.h>
#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <limits>

#include "src/operator/scaled_dot_product_attention_op.h"
#include "util/bf16.h"
namespace {
using namespace feather;
using namespace feather::operators;
std::shared_ptr<Tensor> F(const std::vector<float>& v, const std::vector<int64_t>& s) {
    auto t = std::make_shared<Tensor>();
    t->Assign<float>(v, s);
    return t;
}
std::vector<DeviceType> Devices() {
    std::vector<DeviceType> devices{DeviceType::COMMON};
#ifdef FEATHER_WITH_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        kernel::EnsureCudaScaledDotProductAttentionKernelsRegistered();
        devices.push_back(DeviceType::CUDA);
    }
#endif
    return devices;
}
TEST(ScaledDotProductAttention, GqaFullCausalAndBroadcastMask) {
    for (auto device : Devices())
        for (bool causal : {false, true}) {
            ScaledDotProductAttentionParam p;
            p.q = F({1, 2, 3, 4, -1, 1, 2, -2}, {1, 2, 2, 2});
            p.k = F({1, 0, 0, 1, 1, 1}, {1, 1, 3, 2});
            p.v = F({2, -1, 4, 3, 1, 7}, {1, 1, 3, 2});
            p.mask = F({0, -2, -std::numeric_limits<float>::infinity()}, {1, 1, 1, 3});
            p.out = std::make_shared<Tensor>();
            p.causal = causal;
            ScaledDotProductAttentionOp op(p);
            ASSERT_EQ(op.InferOutputShapes(), 0);
            kernel::EnsureScaledDotProductAttentionKernelsRegistered();
            auto k = KernelDispatcher::instance().create(device, DataType::FP32, "ScaledDotProductAttention");
            ASSERT_NE(k, nullptr);
            op.AttachKernel(std::move(k));
            ASSERT_EQ(op.Run(), 0);
            for (int h = 0; h < 2; ++h)
                for (int i = 0; i < 2; ++i) {
                    double weights[3], sum = 0;
                    for (int j = 0; j < 3; ++j) {
                        double dot = 0;
                        for (int d = 0; d < 2; ++d)
                            dot += p.q->data<float>()[h * 4 + i * 2 + d] * p.k->data<float>()[j * 2 + d];
                        weights[j] = (causal && j > i) ? 0 : std::exp(dot / std::sqrt(2.) + p.mask->data<float>()[j]);
                        sum += weights[j];
                    }
                    for (int d = 0; d < 2; ++d) {
                        double ref = 0;
                        for (int j = 0; j < 3; ++j) ref += weights[j] * p.v->data<float>()[j * 2 + d] / sum;
                        EXPECT_NEAR(p.out->data<float>()[h * 4 + i * 2 + d], ref, 1e-5);
                    }
                }
        }
}
TEST(ScaledDotProductAttention, StableAndFullyMasked) {
    for (auto device : Devices()) {
        ScaledDotProductAttentionParam p;
        p.q = F({1000}, {1, 1, 1, 1});
        p.k = F({1000, 999}, {1, 1, 2, 1});
        p.v = F({2, 7}, {1, 1, 2, 1});
        p.out = std::make_shared<Tensor>();
        ScaledDotProductAttentionOp op(p);
        kernel::EnsureScaledDotProductAttentionKernelsRegistered();
        op.AttachKernel(KernelDispatcher::instance().create(device, DataType::FP32, "ScaledDotProductAttention"));
        ASSERT_EQ(op.Run(), 0);
        EXPECT_FLOAT_EQ(p.out->data<float>()[0], 2);
        p.mask = F({-INFINITY, -INFINITY}, {1, 1, 1, 2});
        ScaledDotProductAttentionOp masked(p);
        masked.AttachKernel(KernelDispatcher::instance().create(device, DataType::FP32, "ScaledDotProductAttention"));
        ASSERT_EQ(masked.Run(), 0);
        EXPECT_FLOAT_EQ(p.out->data<float>()[0], 0);
    }
}
TEST(ScaledDotProductAttention, RejectsBadDescriptors) {
    ScaledDotProductAttentionParam p;
    p.q = F({1}, {1, 1, 1, 1});
    p.k = p.q;
    p.v = p.q;
    p.out = std::make_shared<Tensor>();
    EXPECT_TRUE(ValidateScaledDotProductAttention(p, false));
    p.out = p.q;
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.out = std::make_shared<Tensor>();
    p.out->set_data_type(DataType::BF16);
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.out->set_data_type(DataType::UNKNOWN);
    p.scale = INFINITY;
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.scale = 1;
    p.mask = F({0}, {1});
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.mask = nullptr;
    auto original = p.k;
    p.k = F({1, 2}, {1, 2, 1, 1});
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.k = original;
    p.v = F({1, 2}, {2, 1, 1, 1});
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.v = p.q;
    p.mask = F({0, 0}, {1, 1, 1, 2});
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.mask = nullptr;
    p.k = std::make_shared<Tensor>();
    p.k->Resize({1, 1, std::numeric_limits<int64_t>::max(), 2});
    p.k->set_data_type(DataType::FP32);
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
    p.k = original;
    p.out = std::make_shared<Tensor>(*p.q);
    EXPECT_FALSE(ValidateScaledDotProductAttention(p, false));
}
}  // namespace

#include "core/operator_registry.h"
#include "util/fp16.h"
namespace {
std::shared_ptr<Tensor> Typed(const std::vector<float>& v, const std::vector<int64_t>& dims, DataType dtype) {
    if (dtype == DataType::FP32) return F(v, dims);
    auto t = std::make_shared<Tensor>();
    if (dtype == DataType::BF16) {
        std::vector<BFloat16> a;
        for (float x : v) a.push_back({FloatToBFloat16(x)});
        t->Assign<BFloat16>(a, dims);
    } else {
        std::vector<uint16_t> a;
        for (float x : v) a.push_back(FloatToHalf(x));
        t->Assign<uint16_t>(a, dims);
        t->set_data_type(DataType::FP16);
    }
    return t;
}
float Value(const Tensor& t, int i) {
    if (t.data_type() == DataType::FP32) return t.data<float>()[i];
    if (t.data_type() == DataType::BF16) return BFloat16ToFloat(t.data<BFloat16>()[i].bits);
    return HalfToFloat(t.data<uint16_t>()[i]);
}
void CheckTypes(DeviceType device) {
    for (auto dtype : {DataType::FP32, DataType::FP16, DataType::BF16}) {
        // K=4097 exceeds typical per-row shared-memory score budgets. Zero logits
        // yield an exact independently known average across many online tiles.
        std::vector<float> keys(4097, 0), values(4097);
        for (int i = 0; i < 4097; ++i) values[i] = i % 3;
        ScaledDotProductAttentionParam p;
        p.q = Typed({0, 0}, {1, 2, 1, 1}, dtype);
        p.k = Typed(keys, {1, 1, 4097, 1}, dtype);
        p.v = Typed(values, {1, 1, 4097, 1}, dtype);
        p.out = std::make_shared<Tensor>();
        ScaledDotProductAttentionOp op(p);
        auto kernel = KernelDispatcher::instance().create(device, dtype, "ScaledDotProductAttention");
        ASSERT_NE(kernel, nullptr);
        op.AttachKernel(std::move(kernel));
        ASSERT_EQ(op.Run(), 0);
        const float ref = 4096.0f / 4097.0f;
        for (int i = 0; i < 2; ++i) EXPECT_NEAR(Value(*p.out, i), ref, dtype == DataType::FP32 ? 1e-5 : 0.01);
    }
}
TEST(ScaledDotProductAttention, CommonAllDtypesLongKeys) {
    kernel::EnsureScaledDotProductAttentionKernelsRegistered();
    CheckTypes(DeviceType::COMMON);
}
#ifdef FEATHER_WITH_CUDA
TEST(ScaledDotProductAttention, CudaAllDtypesLongKeys) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    kernel::EnsureCudaScaledDotProductAttentionKernelsRegistered();
    CheckTypes(DeviceType::CUDA);
}
#endif
TEST(ScaledDotProductAttention, RejectsMalformedAttributes) {
    EnsureScaledDotProductAttentionOperatorRegistered();
    OperatorRegistry::TensorMap tensors{{"q", F({1}, {1, 1, 1, 1})},
                                        {"k", F({1}, {1, 1, 1, 1})},
                                        {"v", F({1}, {1, 1, 1, 1})},
                                        {"o", std::make_shared<Tensor>()}};
    model::NodeDesc n;
    n.op_type = "ScaledDotProductAttention";
    n.inputs = {"q", "k", "v"};
    n.outputs = {"o"};
    OperatorRegistry::BuildContext c;
    c.device = DeviceType::COMMON;
    for (const model::AttributeValue& bad :
         std::vector<model::AttributeValue>{int64_t(2), float(1), std::string("true")}) {
        n.attributes = {{"causal", bad}};
        EXPECT_EQ(OperatorRegistry::instance().Create(n, tensors, c), nullptr);
    }
    n.attributes = {{"scale", std::numeric_limits<float>::quiet_NaN()}};
    EXPECT_EQ(OperatorRegistry::instance().Create(n, tensors, c), nullptr);
    n.attributes = {{"unexpected", int64_t(1)}};
    EXPECT_EQ(OperatorRegistry::instance().Create(n, tensors, c), nullptr);
    n.attributes = {{"causal", int64_t(0)}};
    EXPECT_NE(OperatorRegistry::instance().Create(n, tensors, c), nullptr);
}
}  // namespace
namespace {
TEST(ScaledDotProductAttention, MaskedTilesAndMultipleValueTiles) {
    for (auto device : Devices()) {
        std::vector<float> mask(65, -INFINITY);
        mask.back() = 0;
        std::vector<float> values(2 * 65 * 129);
        for (int h = 0; h < 2; ++h)
            for (int k = 0; k < 65; ++k)
                for (int d = 0; d < 129; ++d) values[(h * 65 + k) * 129 + d] = h * 10 + d * 0.125f + k;
        ScaledDotProductAttentionParam p;
        p.q = F(std::vector<float>(2 * 4 * 3 * 5, 1), {2, 4, 3, 5});
        p.k = F(std::vector<float>(2 * 1 * 65 * 5, 1), {2, 1, 65, 5});
        p.v = F(values, {2, 1, 65, 129});
        p.mask = F(mask, {1, 1, 1, 65});
        p.out = std::make_shared<Tensor>();
        p.scale = 0.25f;
        ScaledDotProductAttentionOp op(p);
        op.AttachKernel(KernelDispatcher::instance().create(device, DataType::FP32, "ScaledDotProductAttention"));
        ASSERT_EQ(op.Run(), 0);
        for (int b = 0; b < 2; ++b)
            for (int h = 0; h < 4; ++h)
                for (int q = 0; q < 3; ++q)
                    for (int d = 0; d < 129; ++d)
                        EXPECT_FLOAT_EQ(p.out->data<float>()[((b * 4 + h) * 3 + q) * 129 + d],
                                        b * 10 + d * 0.125f + 64);
    }
}
}  // namespace
