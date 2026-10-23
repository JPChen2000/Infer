// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "core/execution_plan.h"
#include "core/stateful_session.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace {
std::shared_ptr<Tensor> StateValues(float value, size_t count = 4) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(std::vector<float>(count, value), {static_cast<int64_t>(count)});
    return tensor;
}

void BuildStateGraph(StaticGraph& graph, DeviceType device = DeviceType::COMMON, int64_t sequence = 1, bool relay = false) {
    model::ModelDesc model;
    model.graph.inputs = {"state", "tokens"}; model.graph.outputs = {"next"};
    for (const auto* name : {"state", "tokens", "increment", "next"}) {
        model::ValueDesc value;
        value.tensor.name = name;
        value.tensor.dims = value.tensor.name == "tokens" ? std::vector<int64_t>{sequence} : std::vector<int64_t>{4};
        value.tensor.data_type = DataType::FP32;
        value.constant = value.tensor.name == "increment";
        model.graph.values.push_back(value);
    }
    model::NodeDesc node; node.name = "advance"; node.op_type = "Add";
    node.inputs = {"state", "increment"}; node.outputs = {relay ? "produced" : "next"}; model.graph.nodes.push_back(node);
    if (relay) {
        auto value = model.graph.values.back(); value.tensor.name = "produced"; model.graph.values.push_back(value);
        model::NodeDesc identity; identity.name = "relay"; identity.op_type = "Identity";
        identity.inputs = {"produced"}; identity.outputs = {"next"}; model.graph.nodes.push_back(identity);
    }
    graph.SetKernelDevice(device);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("state", StateValues(0)), 0);
    ASSERT_EQ(graph.SetTensor("tokens", StateValues(0, static_cast<size_t>(sequence))), 0);
    ASSERT_EQ(graph.SetTensor("increment", StateValues(2)), 0);
    ASSERT_EQ(graph.Build(), 0);
}

TEST(shared_state_session_test, ShapePlansKeepExactStateObjectAndStorageAfterSourceDestruction) {
    auto state = StateValues(3);
    const auto storage = state->storage();
    const auto address = state->raw_data();
    std::shared_ptr<const CompiledGraph> first_plan, second_plan;
    {
        StaticGraph first, second; BuildStateGraph(first); BuildStateGraph(second, DeviceType::COMMON, 8);
        ASSERT_TRUE(CompiledGraph::Compile(first, &first_plan, {"state"}).ok());
        ASSERT_TRUE(CompiledGraph::Compile(second, &second_plan, {"state"}).ok());
    }
    SessionOptions options; options.reuse_memory = false; options.mutable_state["state"] = state;
    std::unique_ptr<ExecutionSession> first, second;
    ASSERT_TRUE(first_plan->CreateSession(&first, options).ok());
    ASSERT_TRUE(second_plan->CreateSession(&second, options).ok());
    EXPECT_EQ(first->runtime().GetTensor("state"), state);
    EXPECT_EQ(second->runtime().GetTensor("state"), state);
    ASSERT_TRUE(first->Run().ok());
    ASSERT_TRUE(ApplyStateBinding(first->runtime(), {"state", "next", -1, true}, DeviceType::COMMON).ok());
    EXPECT_EQ(state->storage(), storage); EXPECT_EQ(state->raw_data(), address);
    EXPECT_FLOAT_EQ(state->data<float>()[0], 5);
    first.reset(); first_plan.reset();
    ASSERT_TRUE(second->Run().ok());
    StatefulSession recurrent(second->runtime());
    ASSERT_TRUE(recurrent.Bind({"state", "next", -1, true}).ok());
    ASSERT_TRUE(recurrent.Advance().ok());
    EXPECT_FLOAT_EQ(state->data<float>()[0], 7);
    ASSERT_TRUE(recurrent.Reset().ok());
    EXPECT_FLOAT_EQ(state->data<float>()[0], 0);
    EXPECT_EQ(state->storage(), storage); EXPECT_EQ(state->raw_data(), address);
}

TEST(shared_state_session_test, DescriptorOnlyPlansRequireExternalStateAndValidateBinding) {
    StaticGraph graph; BuildStateGraph(graph);
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled, {"state"}).ok());
    std::unique_ptr<ExecutionSession> session;
    EXPECT_FALSE(compiled->CreateSession(&session).ok());
    SessionOptions options;
    options.mutable_state["missing"] = StateValues(0);
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    options.mutable_state.clear(); options.mutable_state["next"] = StateValues(0);
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    options.mutable_state.clear(); options.mutable_state["increment"] = StateValues(0);
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    for (int invalid = 0; invalid != 7; ++invalid) {
        auto tensor = StateValues(0);
        switch (invalid) {
            case 0: tensor->Resize(std::vector<int64_t>{2, 2}); break;
            case 1: tensor->set_data_type(DataType::INT32); break;
            case 2: tensor->set_layout(DataLayout::NHWC); break;
            case 3: tensor->set_immutable(true); break;
            case 4: {
                QuantizationParams quantization; quantization.enabled = true; quantization.scale = .5f;
                tensor->set_quantization(quantization); break;
            }
            case 5: tensor->ResetStorage(std::make_shared<Storage>(4), 4); break;
            case 6: tensor->clear(); break;
        }
        options.mutable_state.clear(); options.mutable_state["state"] = tensor;
        EXPECT_FALSE(compiled->CreateSession(&session, options).ok()) << invalid;
        EXPECT_EQ(session, nullptr);
    }
    options.mutable_state.clear(); options.mutable_state["state"] = StateValues(11);
    ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    EXPECT_FLOAT_EQ(options.mutable_state["state"]->data<float>()[0], 11);
    EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled, {"increment"}).ok());
    EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled, {"next"}).ok());
}

TEST(shared_state_session_test, OverlappingExternalInputsAreRejectedEvenWithSeparateStorageWrappers) {
    StaticGraph graph; BuildStateGraph(graph, DeviceType::COMMON, 4);
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled, {"state", "tokens"}).ok());
    auto storage = StateValues(1, 8);
    auto first = StateValues(0), second = StateValues(0);
    first->ResetBuffer(std::make_shared<Buffer>(storage->mutable_data(storage->memory_size()), 4 * sizeof(float), storage), 4 * sizeof(float));
    second->ResetBuffer(std::make_shared<Buffer>(static_cast<float*>(storage->mutable_data(storage->memory_size())) + 2,
        4 * sizeof(float), storage), 4 * sizeof(float));
    SessionOptions options; options.mutable_state = {{"state", first}, {"tokens", second}};
    std::unique_ptr<ExecutionSession> session;
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    second->ResetBuffer(std::make_shared<Buffer>(static_cast<float*>(storage->mutable_data(storage->memory_size())) + 4,
        4 * sizeof(float), storage), 4 * sizeof(float));
    ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    EXPECT_EQ(session->runtime().GetTensor("state"), first);
    EXPECT_EQ(session->runtime().GetTensor("tokens"), second);
}

TEST(shared_state_session_test, ExactTensorAliasesRemainOneObjectAndSeparateViewsAreRejected) {
    StaticGraph graph; BuildStateGraph(graph, DeviceType::COMMON, 4);
    ASSERT_EQ(graph.SetTensor("tokens", graph.GetTensor("state")), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled, {"state"}).ok());
    auto state = StateValues(3);
    SessionOptions options; options.mutable_state["state"] = state;
    std::unique_ptr<ExecutionSession> session;
    ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    EXPECT_EQ(session->runtime().GetTensor("tokens"), state);
    options.mutable_state["tokens"] = StateValues(4);
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    auto view = StateValues(0); view->ShareDataWith(*graph.GetTensor("state"));
    ASSERT_EQ(graph.SetTensor("tokens", view), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled, {"state"}).ok());
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    options.mutable_state.erase("tokens");
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
}

TEST(shared_state_session_test, StableCopyPreservesOffsetsAndCopiesLogicalExtentOnly) {
    RuntimeGraph graph;
    auto backing = StateValues(9, 12);
    auto input = StateValues(0), output = StateValues(0);
    input->ResetStorage(backing->storage(), 6 * sizeof(float), sizeof(float));
    output->ResetStorage(backing->storage(), 4 * sizeof(float), 6 * sizeof(float));
    for (size_t i = 6; i < 10; ++i) backing->mutable_data<float>()[i] = 2;
    ASSERT_EQ(graph.SetTensor("state", input), 0); ASSERT_EQ(graph.SetTensor("next", output), 0);
    const auto address = input->raw_data();
    ASSERT_TRUE(ApplyStateBinding(graph, {"state", "next", -1, true}, DeviceType::COMMON).ok());
    EXPECT_EQ(input->raw_data(), address);
    for (size_t i = 1; i < 5; ++i) EXPECT_FLOAT_EQ(backing->data<float>()[i], 2);
    EXPECT_FLOAT_EQ(backing->data<float>()[0], 9); EXPECT_FLOAT_EQ(backing->data<float>()[5], 9);
}

#ifdef FEATHER_WITH_CUDA
TEST(shared_state_session_test, SharedCudaContextPreservesDeviceStateAndWeightsWhenOnePlanIsDestroyed) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    std::unique_ptr<ExecutionSession> first, second;
    std::shared_ptr<const CompiledGraph> compiled;
    {
        StaticGraph graph; BuildStateGraph(graph, DeviceType::CUDA);
        ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    }
    ASSERT_TRUE(compiled->CreateSession(&first).ok());
    const auto context = first->runtime().CudaContext();
    auto state = first->runtime().GetTensor("state");
    const auto storage = state->storage();
    SessionOptions options; options.mutable_state["state"] = state;
    options.backend_factory = [context] { return CreateBackendContext(context); };
    options.immutable_weight_source = first.get();
    ASSERT_TRUE(compiled->CreateSession(&second, options).ok());
    EXPECT_EQ(first->runtime().CudaContext(), second->runtime().CudaContext());
    kernel::cuda_detail::ExecutionContextScope scope(context);
    kernel::cuda_detail::DeferredHostSyncScope deferred;
    auto step = [](ExecutionSession& session) {
        auto status = session.Run();
        return status.ok() ? ApplyStateBinding(session.runtime(), {"state", "next", -1, true}, DeviceType::CUDA) : status;
    };
    ASSERT_TRUE(step(*first).ok());
    const auto allocation = first->runtime().Backend().DeviceStorage(*state);
    ASSERT_NE(allocation, nullptr);
    ASSERT_TRUE(step(*second).ok());
    EXPECT_EQ(second->runtime().Backend().DeviceStorage(*state), allocation);
    first.reset();
    ASSERT_TRUE(step(*second).ok());
    ASSERT_TRUE(second->runtime().Backend().ReadHost(*state).ok());
    EXPECT_FLOAT_EQ(state->data<float>()[0], 6);
    EXPECT_EQ(state->storage(), storage);
    EXPECT_EQ(second->runtime().Backend().DeviceStorage(*state), allocation);
    std::shared_ptr<Tensor> result;
    ASSERT_TRUE(second->ReadOutput("next", &result).ok()); EXPECT_FLOAT_EQ(result->data<float>()[0], 6);
}

TEST(shared_state_session_test, StableCudaUpdateCopiesRetainedIdentityOutputAfterProducerRelease) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    StaticGraph graph; BuildStateGraph(graph, DeviceType::CUDA, 1, true);
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled, {"state"}).ok());
    auto state = StateValues(0);
    SessionOptions options; options.mutable_state["state"] = state;
    std::unique_ptr<ExecutionSession> session;
    ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    kernel::cuda_detail::ExecutionContextScope scope(session->runtime().CudaContext());
    kernel::cuda_detail::DeferredHostSyncScope deferred;
    for (int step = 1; step <= 3; ++step) {
        ASSERT_TRUE(session->Run().ok());
        EXPECT_EQ(session->runtime().Backend().DeviceStorage(*session->runtime().GetTensor("produced")), nullptr);
        ASSERT_TRUE(ApplyStateBinding(session->runtime(), {"state", "next", -1, true}, DeviceType::CUDA).ok());
        ASSERT_TRUE(session->runtime().Backend().ReadHost(*state).ok());
        EXPECT_FLOAT_EQ(state->data<float>()[0], step * 2);
    }
}

TEST(shared_state_session_test, CapturedTemporaryAllocationsStayOutOfOtherGraphsAndReturnAfterReset) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    auto context = kernel::cuda_detail::CreateExecutionContext();
    RuntimeGraph owner(CreateBackendContext(context));
    kernel::cuda_detail::ExecutionContextScope scope(context);
    kernel::cuda_detail::DeferredHostSyncScope deferred;
    void* warmed = nullptr;
    ASSERT_EQ(kernel::cuda_detail::AcquireTemporaryDeviceBuffer(sizeof(float), &warmed), 0);
    kernel::cuda_detail::ReleaseTemporaryDeviceBuffer(warmed, sizeof(float));
    StaticGraph graph; BuildStateGraph(graph, DeviceType::CUDA);
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled, {"state"}).ok());
    SessionOptions options; options.mutable_state["state"] = StateValues(5);
    options.backend_factory = [context] { return CreateBackendContext(context); };
    std::unique_ptr<ExecutionSession> other;
    ASSERT_TRUE(compiled->CreateSession(&other, options).ok());
    size_t released_pool_blocks = 0;
    for (int cycle = 0; cycle != 4; ++cycle) {
        void* captured = nullptr;
        std::unique_ptr<ExecutionCapture> executable;
        auto callback = [&]() -> Status {
            if (kernel::cuda_detail::AcquireTemporaryDeviceBuffer(sizeof(float), &captured) != 0)
                return Status::Error(StatusCode::kExecutionFailed, "temporary allocation failed");
            const auto status = cudaMemsetAsync(captured, 0, sizeof(float), kernel::cuda_detail::InferenceStream());
            kernel::cuda_detail::ReleaseTemporaryDeviceBuffer(captured, sizeof(float));
            return status == cudaSuccess ? Status::Ok() : Status::Error(StatusCode::kExecutionFailed, "memset failed");
        };
        const auto status = owner.Backend().Capture(callback, &executable);
        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_EQ(captured, warmed);  // Reset cycles recycle the same allocation.
        ASSERT_TRUE(other->Run().ok());
        auto state_device = other->runtime().Backend().DeviceStorage(*options.mutable_state["state"]);
        ASSERT_NE(state_device, nullptr); EXPECT_NE(state_device->data(), captured);
        ASSERT_TRUE(executable->Launch().ok());
        ASSERT_TRUE(other->Run().ok());
        std::shared_ptr<Tensor> output;
        ASSERT_TRUE(other->ReadOutput("next", &output).ok());
        EXPECT_FLOAT_EQ(output->data<float>()[0], 7);
        const auto blocks_while_pinned = kernel::cuda_detail::GetTensorCacheStats().free_block_count;
        executable.reset();
        const auto blocks_after_reset = kernel::cuda_detail::GetTensorCacheStats().free_block_count;
        EXPECT_EQ(blocks_after_reset, blocks_while_pinned + 1);
        if (cycle) EXPECT_EQ(blocks_after_reset, released_pool_blocks);
        released_pool_blocks = blocks_after_reset;
    }
    // A failed capture abandons its recording without leaving allocation pins.
    std::unique_ptr<ExecutionCapture> failed;
    const auto status = owner.Backend().Capture([&]() -> Status {
        void* temporary = nullptr;
        if (kernel::cuda_detail::AcquireTemporaryDeviceBuffer(sizeof(float), &temporary) != 0)
            return Status::Error(StatusCode::kExecutionFailed, "temporary allocation failed");
        (void)cudaMemsetAsync(temporary, 0, sizeof(float), kernel::cuda_detail::InferenceStream());
        kernel::cuda_detail::ReleaseTemporaryDeviceBuffer(temporary, sizeof(float));
        return Status::Error(StatusCode::kExecutionFailed, "intentional capture failure");
    }, &failed);
    EXPECT_FALSE(status.ok()); EXPECT_EQ(failed, nullptr);
    EXPECT_EQ(kernel::cuda_detail::GetTensorCacheStats().free_block_count, released_pool_blocks);
}

TEST(shared_state_session_test, FailedCaptureRestoresHostUploadCoherenceAndDeviceAliasesBeforeFallback) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    auto context = kernel::cuda_detail::CreateExecutionContext();
    RuntimeGraph graph(CreateBackendContext(context));
    auto input = StateValues(1), alias = StateValues(9);
    ASSERT_EQ(graph.SetTensor("input", input), 0); ASSERT_EQ(graph.SetTensor("alias", alias), 0);
    kernel::cuda_detail::ExecutionContextScope scope(context);
    kernel::cuda_detail::DeferredHostSyncScope deferred;
    void* input_device = nullptr; void* alias_device = nullptr;
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(input.get(), input->memory_size(), input->raw_data(), &input_device), 0);
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(alias.get(), alias->memory_size(), alias->raw_data(), &alias_device), 0);
    ASSERT_EQ(kernel::cuda_detail::SynchronizeInferenceStream(), 0);
    input->mutable_data<float>()[0] = 5;
    graph.Backend().HostWritten(*input);
    const auto pool_before = kernel::cuda_detail::GetTensorCacheStats().free_block_count;
    std::unique_ptr<ExecutionCapture> executable;
    const auto status = graph.Backend().Capture([&]() -> Status {
        void* recorded = nullptr;
        if (kernel::cuda_detail::AcquireTensorDevice(input.get(), input->memory_size(), input->raw_data(), &recorded) != 0 ||
            kernel::cuda_detail::AliasTensorDeviceStorage(input.get(), alias.get(), input->memory_size()) != 0)
            return Status::Error(StatusCode::kExecutionFailed, "capture setup failed");
        return Status::Error(StatusCode::kExecutionFailed, "intentional capture failure");
    }, &executable);
    EXPECT_FALSE(status.ok()); EXPECT_EQ(executable, nullptr);
    EXPECT_EQ(kernel::cuda_detail::GetTensorCacheStats().free_block_count, pool_before);
    auto restored_alias = graph.Backend().DeviceStorage(*alias);
    ASSERT_NE(restored_alias, nullptr); EXPECT_EQ(restored_alias->data(), alias_device);
    void* actual = nullptr;
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(input.get(), input->memory_size(), input->raw_data(), &actual), 0);
    EXPECT_EQ(actual, input_device);
    ASSERT_EQ(kernel::cuda_detail::SynchronizeInferenceStream(), 0);
    float value = 0;
    ASSERT_EQ(cudaMemcpy(&value, actual, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(value, 5);  // The abandoned H2D node never executed.
    ASSERT_EQ(cudaMemcpy(&value, alias_device, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(value, 9);  // Alias metadata from the abandoned graph was undone.
}

TEST(shared_state_session_test, ClearDropsOnlyFinalGraphRegistrationsIncludingPersistentTensorHandles) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    auto context = kernel::cuda_detail::CreateExecutionContext();
    auto first = std::make_unique<RuntimeGraph>(CreateBackendContext(context));
    auto second = std::make_unique<RuntimeGraph>(CreateBackendContext(context));
    auto shared = StateValues(5), private_value = StateValues(7);
    ASSERT_EQ(first->SetTensor("state", shared), 0);
    ASSERT_EQ(first->SetTensor("alias", shared), 0);  // One graph registers a handle once.
    ASSERT_EQ(first->SetTensor("private", private_value), 0);
    ASSERT_EQ(second->SetTensor("state", shared), 0);
    kernel::cuda_detail::ExecutionContextScope scope(context);
    void* shared_device = nullptr; void* private_device = nullptr;
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(shared.get(), shared->memory_size(), shared->raw_data(), &shared_device), 0);
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(private_value.get(), private_value->memory_size(), private_value->raw_data(), &private_device), 0);
    first->Backend().MarkPersistent(*shared, true); first->Backend().MarkPersistent(*private_value, true);
    first->Clear();
    EXPECT_EQ(kernel::cuda_detail::GetTensorCacheStats().active_tensor_count, 1);
    auto device_storage = second->Backend().DeviceStorage(*shared);
    ASSERT_NE(device_storage, nullptr); EXPECT_EQ(device_storage->data(), shared_device);
    EXPECT_EQ(second->Backend().DeviceStorage(*private_value), nullptr);
    second.reset();
    EXPECT_EQ(kernel::cuda_detail::GetTensorCacheStats().active_tensor_count, 0);
    float value = 0;
    ASSERT_EQ(cudaMemcpy(&value, device_storage->data(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(value, 5);
}
#endif
}  // namespace
}  // namespace feather
