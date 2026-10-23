// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/operator_registry.h"
#include "src/kernel/qwen_gated_delta.h"
#include "src/kernel/qwen_gated_delta_sequence.h"
#include "src/operator/params.h"
#include "src/operator/qwen_gated_delta_sequence_op.h"

namespace {
using namespace feather;
using operators::QwenGatedDeltaSequenceParam;

std::shared_ptr<Tensor> Values(const std::vector<int64_t>& dims, int seed = 0) {
    size_t count = 0;
    EXPECT_TRUE(CheckedElementCount(dims, &count));
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((static_cast<int>(i % 23) + seed) % 23 - 11) / 64.0f;
    }
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

QwenGatedDeltaSequenceParam Inputs(int64_t sequence, int64_t heads = 2, int64_t key = 3,
                                  int64_t value = 5) {
    QwenGatedDeltaSequenceParam p{};
    p.state = Values({1, heads, key, value}, 3); // Deliberately nonzero initial state.
    p.k = Values({1, sequence, heads, key}, 7);
    p.q = Values({1, sequence, heads, key}, 11);
    p.v = Values({1, sequence, heads, value}, 17);
    p.beta = Values({1, sequence, heads});
    p.decay = Values({1, sequence, heads});
    for (int64_t i = 0; i < sequence * heads; ++i) {
        p.beta->mutable_data<float>()[i] = 0.1f + 0.03f * (i % 7);
        p.decay->mutable_data<float>()[i] = 0.8f + 0.02f * (i % 9);
    }
    p.next_state = Values({1, heads, key, value});
    p.out = Values({1, sequence, heads, value});
    return p;
}

std::shared_ptr<Tensor> Token(const std::shared_ptr<Tensor>& source, int64_t t,
                             const std::vector<int64_t>& shape) {
    size_t count = 0;
    EXPECT_TRUE(CheckedElementCount(shape, &count));
    auto result = std::make_shared<Tensor>();
    const float* start = source->data<float>() + static_cast<size_t>(t) * count;
    result->Assign<float>(std::vector<float>(start, start + count), shape);
    return result;
}

class QwenGatedDeltaSequenceTest : public testing::TestWithParam<DeviceType> {
 protected:
    void SetUp() override {
        kernel::EnsureQwenGatedDeltaKernelsRegistered();
        kernel::EnsureQwenGatedDeltaSequenceKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
        if (GetParam() == DeviceType::CUDA) {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
        }
#endif
    }
    std::unique_ptr<KernelBase> SequenceKernel() {
        return KernelDispatcher::instance().create(GetParam(), DataType::FP32, "QwenGatedDeltaSequence");
    }
};

// Catches reversed time order, reading q before updating state, dropped prefixes,
// wrong token/head strides, and incomplete value lanes in the CUDA block.
TEST_P(QwenGatedDeltaSequenceTest, MatchesRepeatedCombinedStepsWithNonzeroPrefix) {
    for (const auto& dims : std::vector<std::vector<int64_t>>{{1, 2, 3, 5}, {3, 2, 3, 5},
                                                           {17, 2, 64, 128}, {3, 2, 5, 513},
                                                           {257, 2, 7, 9}}) {
        const int64_t sequence = dims[0], heads = dims[1], key = dims[2], value = dims[3];
        SCOPED_TRACE(testing::Message() << "T=" << sequence << " K=" << key << " V=" << value);
        auto p = Inputs(sequence, heads, key, value);
        const std::vector<float> original(p.state->data<float>(), p.state->data<float>() + p.state->numel());
        auto state = p.state;
        std::vector<float> expected;
        for (int64_t t = 0; t < sequence; ++t) {
            operators::QwenGatedDeltaParam one{};
            one.state = state;
            one.k = Token(p.k, t, {1, heads, key});
            one.q = Token(p.q, t, {1, heads, key});
            one.v = Token(p.v, t, {1, heads, value});
            one.beta = Token(p.beta, t, {1, heads, 1});
            one.decay = Token(p.decay, t, {1, heads, 1, 1});
            one.next_state = Values({1, heads, key, value});
            one.out = Values({1, heads, value});
            auto serial = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP32,
                                                               "QwenGatedDelta");
            ASSERT_NE(serial, nullptr);
            serial->SetParam(&one);
            ASSERT_EQ(serial->compute(), 0);
            expected.insert(expected.end(), one.out->data<float>(), one.out->data<float>() + one.out->numel());
            state = one.next_state;
        }
        auto sequence_kernel = SequenceKernel();
        ASSERT_NE(sequence_kernel, nullptr);
        sequence_kernel->SetParam(&p);
        ASSERT_EQ(sequence_kernel->compute(), 0);
        for (int64_t i = 0; i < state->numel(); ++i) {
            EXPECT_NEAR(p.next_state->data<float>()[i], state->data<float>()[i], 2e-5f) << i;
        }
        for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(p.out->data<float>()[i], expected[i], 2e-5f) << i;
        EXPECT_TRUE(std::equal(original.begin(), original.end(), p.state->data<float>()));
    }
}

TEST_P(QwenGatedDeltaSequenceTest, RejectsMalformedDescriptorsAndOverlappingBuffers) {
    auto sequence_kernel = SequenceKernel();
    ASSERT_NE(sequence_kernel, nullptr);
    auto reject = [&](QwenGatedDeltaSequenceParam p) {
        sequence_kernel->SetParam(&p);
        EXPECT_NE(sequence_kernel->compute(), 0);
        operators::QwenGatedDeltaSequenceOp op("invalid_sequence", p);
        EXPECT_NE(op.CheckShape(), 0);
    };
    auto p = Inputs(3);
    p.k->Resize({1, 2, 3, 3}); // Same element count, wrong sequence/head order.
    reject(p);
    p = Inputs(3); p.q->Resize({1, 3, 6}); reject(p);
    p = Inputs(3); p.beta->Resize({1, 3, 2, 1}); reject(p);
    p = Inputs(3); p.decay->Resize({1, 0, 2}); reject(p);
    p = Inputs(3); p.state->Resize({2, 1, 3, 5}); reject(p);
    p = Inputs(3); p.k->set_data_type(DataType::BF16); reject(p);
    p = Inputs(3); p.next_state = p.state; reject(p);
    p = Inputs(3); p.out->ShareDataWith(*p.v); reject(p);
    p = Inputs(3); p.next_state->ShareDataWith(*p.state); reject(p);
    p = Inputs(3); p.q.reset(); reject(p);
    p = Inputs(3);
    p.state->Resize({1, 2, std::numeric_limits<int64_t>::max(), 5});
    p.k->Resize({1, 3, 2, std::numeric_limits<int64_t>::max()});
    p.q->Resize(p.k->dims().data());
    reject(p);
    p = Inputs(3);
    const auto huge = std::numeric_limits<int64_t>::max();
    p.k->Resize({1, huge, 2, 3}); p.q->Resize(p.k->dims().data());
    p.v->Resize({1, huge, 2, 5});
    p.beta->Resize({1, huge, 2}); p.decay->Resize(p.beta->dims().data());
    reject(p);
}

TEST_P(QwenGatedDeltaSequenceTest, RejectsUnallocatedOrWrongOutputShapeAndType) {
    auto sequence_kernel = SequenceKernel();
    ASSERT_NE(sequence_kernel, nullptr);
    auto p = Inputs(3);
    p.out->Resize({1, 2, 3, 5}); // Same count is insufficient for an output descriptor.
    sequence_kernel->SetParam(&p); EXPECT_NE(sequence_kernel->compute(), 0);
    p = Inputs(3); p.next_state->set_data_type(DataType::BF16);
    sequence_kernel->SetParam(&p); EXPECT_NE(sequence_kernel->compute(), 0);
    p = Inputs(3); p.k = std::make_shared<Tensor>(); p.k->Resize({1, 3, 2, 3});
    p.k->set_data_type(DataType::FP32);
    sequence_kernel->SetParam(&p); EXPECT_NE(sequence_kernel->compute(), 0);
    p = Inputs(3); p.out = Values({1}); p.out->Resize({1, 3, 2, 5});
    sequence_kernel->SetParam(&p); EXPECT_NE(sequence_kernel->compute(), 0);
}

// Literal two-token result checks the recurrence independently of either kernel.
// s=2 -> 2*.5 + .5*(3 - 2*(.5*.5))*.25 = 1.3125;
// then s=1.3125 + (-1)*(1 - 1.3125*(-1))*.5 = .15625.
TEST_P(QwenGatedDeltaSequenceTest, ReadsEachUpdatedStateBeforeAdvancingTime) {
    auto p = Inputs(2, 1, 1, 1);
    p.state->Assign<float>({2.0f}, {1, 1, 1, 1});
    p.k->Assign<float>({0.5f, -1.0f}, {1, 2, 1, 1});
    p.q->Assign<float>({2.0f, 4.0f}, {1, 2, 1, 1});
    p.v->Assign<float>({3.0f, 1.0f}, {1, 2, 1, 1});
    p.beta->Assign<float>({0.25f, 0.5f}, {1, 2, 1});
    p.decay->Assign<float>({0.5f, 1.0f}, {1, 2, 1});
    auto sequence_kernel = SequenceKernel();
    ASSERT_NE(sequence_kernel, nullptr);
    sequence_kernel->SetParam(&p);
    ASSERT_EQ(sequence_kernel->compute(), 0);
    EXPECT_FLOAT_EQ(p.out->data<float>()[0], 2.625f);
    EXPECT_FLOAT_EQ(p.out->data<float>()[1], 0.625f);
    EXPECT_FLOAT_EQ(p.next_state->data<float>()[0], 0.15625f);
}

INSTANTIATE_TEST_SUITE_P(CommonAndCuda, QwenGatedDeltaSequenceTest, testing::Values(DeviceType::COMMON
#ifdef FEATHER_WITH_CUDA
    , DeviceType::CUDA
#endif
));

// Catches allocating a new Tensor handle after the builder/kernel captured the old one.
TEST(QwenGatedDeltaSequenceOperatorTest, InfersAndRunsIntoOriginalOutputHandles) {
    kernel::EnsureQwenGatedDeltaSequenceKernelsRegistered();
    operators::EnsureQwenGatedDeltaSequenceOperatorsRegistered();
    auto p = Inputs(3);
    p.next_state = std::make_shared<Tensor>();
    p.out = std::make_shared<Tensor>();
    operators::QwenGatedDeltaSequenceOp op("sequence", p);
    ASSERT_EQ(op.InferOutputShapes(), 0);
    EXPECT_EQ(op.outputs()[0].get(), p.next_state.get());
    EXPECT_EQ(op.outputs()[1].get(), p.out.get());
    EXPECT_EQ(p.out->dims().data(), (std::vector<int64_t>{1, 3, 2, 5}));
    auto compute = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP32,
                                                       "QwenGatedDeltaSequence");
    ASSERT_NE(compute, nullptr);
    op.AttachKernel(std::move(compute));
    ASSERT_EQ(op.Run(), 0);
    EXPECT_EQ(p.out->data_type(), DataType::FP32);
}

TEST(QwenGatedDeltaSequenceOperatorTest, RebindsEveryTensorForAnExecutionSession) {
    kernel::EnsureQwenGatedDeltaSequenceKernelsRegistered();
    auto original = Inputs(3);
    auto rebound = Inputs(3);
    auto compute = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP32,
                                                       "QwenGatedDeltaSequence");
    ASSERT_NE(compute, nullptr);
    compute->SetParamOwner(std::make_shared<QwenGatedDeltaSequenceParam>(original));
    auto session_compute = compute->Clone();
    ASSERT_NE(session_compute, nullptr);
    KernelBase::TensorBindings bindings;
    std::vector<std::shared_ptr<Tensor>> original_tensors, rebound_tensors;
    original.VisitTensors([&](std::shared_ptr<Tensor>& tensor) { original_tensors.push_back(tensor); });
    rebound.VisitTensors([&](std::shared_ptr<Tensor>& tensor) { rebound_tensors.push_back(tensor); });
    for (size_t i = 0; i < original_tensors.size(); ++i) bindings.emplace(original_tensors[i].get(), rebound_tensors[i]);
    const auto binder = compute->SnapshotParameters();
    ASSERT_TRUE(binder);
    ASSERT_TRUE(binder(*session_compute, bindings).ok());
    const auto* p = session_compute->ParamAs<QwenGatedDeltaSequenceParam>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->state, rebound.state);
    EXPECT_EQ(p->k, rebound.k);
    EXPECT_EQ(p->v, rebound.v);
    EXPECT_EQ(p->beta, rebound.beta);
    EXPECT_EQ(p->decay, rebound.decay);
    EXPECT_EQ(p->q, rebound.q);
    EXPECT_EQ(p->next_state, rebound.next_state);
    EXPECT_EQ(p->out, rebound.out);
    EXPECT_EQ(session_compute->compute(), 0);
}
} // namespace
