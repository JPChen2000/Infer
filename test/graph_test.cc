#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "core/graph_lowering.h"
#include "core/graph.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "src/kernel/add.h"
#include "src/kernel/mul.h"
#include "src/operator/add_op.h"
#include "src/operator/mul_op.h"
#include "util/threading.h"

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>

#include "src/kernel/cuda/runtime.h"
#endif

using feather::DataType;
using feather::DeviceType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::RuntimeThreadMode;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

namespace {

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

void BuildAddRuntimeWithBackend(DeviceType device, RuntimeGraph* runtime_graph) {
    ModelDesc model;
    model.name = "add_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs"};
    model.graph.outputs = {"out"};

    ValueDesc lhs;
    lhs.tensor.name = "lhs";
    lhs.tensor.dims = {2};
    lhs.tensor.data_type = DataType::FP32;

    ValueDesc rhs;
    rhs.tensor.name = "rhs";
    rhs.tensor.dims = {2};
    rhs.tensor.data_type = DataType::FP32;

    ValueDesc out;
    out.tensor.name = "out";
    out.tensor.dims = {2};
    out.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "add0";
    node.op_type = "Add";
    node.inputs = {"lhs", "rhs"};
    node.outputs = {"out"};

    model.graph.values = {lhs, rhs, out};
    model.graph.nodes = {node};

    auto lhs_tensor = std::make_shared<Tensor>();
    lhs_tensor->Assign<float>({1.0f, 2.0f}, {2});

    auto rhs_tensor = std::make_shared<Tensor>();
    rhs_tensor->Assign<float>({3.0f, 4.0f}, {2});

    StaticGraph static_graph;
    static_graph.SetKernelDevice(device);
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("lhs", lhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("rhs", rhs_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, runtime_graph), 0);
}

#ifdef FEATHER_WITH_CUDA
feather::RuntimeNode MakeBinaryRuntimeNode(const std::string& name, const std::string& op_type,
                                           std::shared_ptr<feather::OpBase> op,
                                           std::unique_ptr<feather::KernelBase> kernel,
                                           std::vector<std::string> inputs,
                                           std::vector<std::string> outputs) {
    feather::RuntimeNode node;
    node.name = name;
    node.op_type = op_type;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    node.owner = std::move(op);
    node.kernel = std::move(kernel);
    node.kernel_device = node.kernel == nullptr ? feather::DeviceType::UNKNOWN : node.kernel->device();
    return node;
}
#endif

}  // namespace

TEST(runtime_graph_test, BuildRuntimeGraphFromStaticGraph) {
    ModelDesc model;
    model.name = "fc_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 4};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc bias;
    bias.tensor.name = "bias";
    bias.tensor.dims = {3, 4};
    bias.tensor.data_type = DataType::FP32;
    bias.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {3, 4};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight", "bias"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, bias, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->Assign<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {3, 4});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("bias", bias_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.OperatorSize(), 1U);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 1U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 4}));

    std::vector<float> expected = {
        12, 15, 18, 21,
        25, 32, 39, 46,
        38, 49, 60, 71,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(static_graph_test, KernelDeviceSelectsCommonBackend) {
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::COMMON, &runtime_graph);

    const auto* node = runtime_graph.GetNode("add0");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->kernel, nullptr);
    EXPECT_EQ(node->kernel_device, DeviceType::COMMON);
    EXPECT_EQ(typeid(*node->kernel),
              typeid(feather::kernel::AddKernel<DeviceType::COMMON, DataType::FP32>));
}

TEST(static_graph_test, KernelDeviceSelectsX86Backend) {
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::X86, &runtime_graph);

    const auto* node = runtime_graph.GetNode("add0");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->kernel, nullptr);
    EXPECT_EQ(node->kernel_device, DeviceType::X86);
    EXPECT_EQ(typeid(*node->kernel),
              typeid(feather::kernel::AddKernel<DeviceType::X86, DataType::FP32>));
}

TEST(static_graph_test, X86LoweringKeepsParallelRuntimeThreadMode) {
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::X86, &runtime_graph);

    EXPECT_EQ(runtime_graph.ThreadMode(), RuntimeThreadMode::kParallelGraph);
}

TEST(static_graph_test, X86LoweringPreservesPreconfiguredSerialRuntimeThreadMode) {
    RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    BuildAddRuntimeWithBackend(DeviceType::X86, &runtime_graph);

    EXPECT_EQ(runtime_graph.ThreadMode(), RuntimeThreadMode::kSerialGraph);
}

TEST(static_graph_test, CudaLoweringUsesSerialRuntimeThreadMode) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    RuntimeGraph runtime_graph;
    BuildAddRuntimeWithBackend(DeviceType::CUDA, &runtime_graph);

    EXPECT_EQ(runtime_graph.ThreadMode(), RuntimeThreadMode::kSerialGraph);
#endif
}

TEST(runtime_graph_test, CudaGraphSynchronizesAroundCommonFallbackNode) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();

    auto lhs = std::make_shared<Tensor>();
    lhs->Assign<float>({1.0f, 2.0f}, {2});
    auto rhs = std::make_shared<Tensor>();
    rhs->Assign<float>({10.0f, 20.0f}, {2});
    auto scale = std::make_shared<Tensor>();
    scale->Assign<float>({2.0f, 3.0f}, {2});
    auto addend = std::make_shared<Tensor>();
    addend->Assign<float>({5.0f, 6.0f}, {2});

    auto cuda_sum = std::make_shared<Tensor>();
    cuda_sum->Assign<float>({-100.0f, -100.0f}, {2});
    auto cpu_product = std::make_shared<Tensor>();
    cpu_product->Assign<float>({-7.0f, -7.0f}, {2});
    auto final = std::make_shared<Tensor>();
    final->Assign<float>({0.0f, 0.0f}, {2});

    void* stale_device_product = nullptr;
    ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                  cpu_product.get(), cpu_product->numel() * sizeof(float), cpu_product->data<float>(),
                  &stale_device_product),
              0);
    ASSERT_NE(stale_device_product, nullptr);

    feather::operators::BinaryParam add0_param{};
    add0_param.lhs = lhs;
    add0_param.rhs = rhs;
    add0_param.out = cuda_sum;
    auto add0_op = std::make_shared<feather::operators::AddOp>("cuda_add0", add0_param);
    auto add0_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(add0_kernel, nullptr);
    add0_op->AttachKernel(std::move(add0_kernel));
    auto add0_detached = add0_op->DetachKernel();

    feather::operators::BinaryParam mul_param{};
    mul_param.lhs = cuda_sum;
    mul_param.rhs = scale;
    mul_param.out = cpu_product;
    auto mul_op = std::make_shared<feather::operators::MulOp>("common_mul", mul_param);
    auto mul_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON, feather::DataType::FP32, "Mul");
    ASSERT_NE(mul_kernel, nullptr);
    mul_op->AttachKernel(std::move(mul_kernel));
    auto mul_detached = mul_op->DetachKernel();

    feather::operators::BinaryParam add1_param{};
    add1_param.lhs = cpu_product;
    add1_param.rhs = addend;
    add1_param.out = final;
    auto add1_op = std::make_shared<feather::operators::AddOp>("cuda_add1", add1_param);
    auto add1_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(add1_kernel, nullptr);
    add1_op->AttachKernel(std::move(add1_kernel));
    auto add1_detached = add1_op->DetachKernel();

    RuntimeGraph graph;
    graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    ASSERT_EQ(graph.SetTensor("lhs", lhs), 0);
    ASSERT_EQ(graph.SetTensor("rhs", rhs), 0);
    ASSERT_EQ(graph.SetTensor("scale", scale), 0);
    ASSERT_EQ(graph.SetTensor("addend", addend), 0);
    ASSERT_EQ(graph.SetTensor("cuda_sum", cuda_sum), 0);
    ASSERT_EQ(graph.SetTensor("cpu_product", cpu_product), 0);
    ASSERT_EQ(graph.SetTensor("final", final), 0);

    graph.AddNode(MakeBinaryRuntimeNode("cuda_add0", "Add", add0_op, std::move(add0_detached), {"lhs", "rhs"},
                                        {"cuda_sum"}));
    graph.AddNode(MakeBinaryRuntimeNode("common_mul", "Mul", mul_op, std::move(mul_detached), {"cuda_sum", "scale"},
                                        {"cpu_product"}));
    graph.AddNode(MakeBinaryRuntimeNode("cuda_add1", "Add", add1_op, std::move(add1_detached),
                                        {"cpu_product", "addend"}, {"final"}));
    ASSERT_EQ(graph.Finalize(), 0);

    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(graph.Run(), 0);
        ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHost(final.get(), final->numel() * sizeof(float),
                                                                 final->mutable_data<float>()),
                  0);
    }

    EXPECT_FLOAT_EQ(final->data<float>()[0], 27.0f);
    EXPECT_FLOAT_EQ(final->data<float>()[1], 72.0f);
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaTensorCacheTracksInPlaceTensorMutation) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1.0f}, {1});

    void* device_ptr = nullptr;
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      input.get(), sizeof(float), input->data<float>(), &device_ptr),
                  0);
        ASSERT_NE(device_ptr, nullptr);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);

        float device_value = 0.0f;
        ASSERT_EQ(cudaMemcpy(&device_value, device_ptr, sizeof(device_value), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_FLOAT_EQ(device_value, 1.0f);

        input->mutable_data<float>()[0] = 2.0f;
        void* updated_device_ptr = nullptr;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      input.get(), sizeof(float), input->data<float>(), &updated_device_ptr),
                  0);
        ASSERT_EQ(updated_device_ptr, device_ptr);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);
        ASSERT_EQ(cudaMemcpy(&device_value, updated_device_ptr, sizeof(device_value), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_FLOAT_EQ(device_value, 2.0f);
    }
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaTensorCacheTracksMutationThroughSharedHostStorage) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();
    auto source = std::make_shared<Tensor>();
    source->Assign<float>({1.0f}, {1});
    auto alias = std::make_shared<Tensor>();
    alias->ShareDataWith(*source);

    void* device_ptr = nullptr;
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      alias.get(), sizeof(float), alias->data<float>(), &device_ptr),
                  0);
        ASSERT_NE(device_ptr, nullptr);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);

        float device_value = 0.0f;
        ASSERT_EQ(cudaMemcpy(&device_value, device_ptr, sizeof(device_value), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_FLOAT_EQ(device_value, 1.0f);

        source->mutable_data<float>()[0] = 2.0f;
        void* updated_device_ptr = nullptr;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      alias.get(), sizeof(float), alias->data<float>(), &updated_device_ptr),
                  0);
        ASSERT_EQ(updated_device_ptr, device_ptr);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);
        ASSERT_EQ(cudaMemcpy(&device_value, updated_device_ptr, sizeof(device_value), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_FLOAT_EQ(device_value, 2.0f);
    }
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaHostMutationIsNotOverwrittenBeforeCommonFallback) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();
    auto value = std::make_shared<Tensor>();
    value->Assign<float>({0.0f}, {1});

    void* device_ptr = nullptr;
    ASSERT_EQ(feather::kernel::cuda_detail::AcquireOutputTensorDevice(value.get(), sizeof(float), &device_ptr), 0);
    ASSERT_NE(device_ptr, nullptr);
    const float device_value = 1.0f;
    ASSERT_EQ(cudaMemcpyAsync(device_ptr, &device_value, sizeof(device_value), cudaMemcpyHostToDevice,
                              feather::kernel::cuda_detail::InferenceStream()),
              cudaSuccess);
    ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);

    value->mutable_data<float>()[0] = 2.0f;
    ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHostIfNeeded(value.get(), sizeof(float), value->raw_data()),
              0);
    EXPECT_FLOAT_EQ(value->data<float>()[0], 2.0f);
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaStateAppendRejectsReleasedOutputDeviceStorage) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();
    auto state = std::make_shared<Tensor>();
    state->Assign<float>({1.0f, 2.0f, 3.0f}, {1, 3, 1});
    auto token = std::make_shared<Tensor>();
    token->Assign<float>({4.0f}, {1, 1, 1});

    void* state_device = nullptr;
    void* token_device = nullptr;
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      state.get(), 3 * sizeof(float), state->data<float>(), &state_device),
                  0);
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      token.get(), sizeof(float), token->data<float>(), &token_device),
                  0);
        ASSERT_NE(state_device, nullptr);
        ASSERT_NE(token_device, nullptr);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);

        feather::kernel::cuda_detail::ReleaseTensorDevice(token.get());
        EXPECT_NE(feather::kernel::cuda_detail::AppendTensorStateOnDevice(state.get(), token.get(), 1), 0);
    }
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaOutputAliasKeepsSourceDeviceStorageAlive) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    ModelDesc model;
    model.name = "qwen_cuda_output_alias_lifetime";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"cache_state", "bias"};
    model.graph.outputs = {"next_cache_state", "consumer_out"};

    auto value = [](const std::string& name) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = {1};
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };
    model.graph.values = {value("cache_state"), value("bias"), value("state_raw"), value("next_cache_state"),
                          value("consumer_out")};

    NodeDesc produce;
    produce.name = "produce_state";
    produce.op_type = "Add";
    produce.inputs = {"cache_state", "bias"};
    produce.outputs = {"state_raw"};

    NodeDesc consume;
    consume.name = "consume_state";
    consume.op_type = "Relu";
    consume.inputs = {"state_raw"};
    consume.outputs = {"consumer_out"};

    NodeDesc relay;
    relay.name = "state_output";
    relay.op_type = "Identity";
    relay.inputs = {"state_raw"};
    relay.outputs = {"next_cache_state"};
    model.graph.nodes = {produce, consume, relay};

    auto cache_state = std::make_shared<Tensor>();
    cache_state->Assign<float>({1.0f}, {1});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({2.0f}, {1});
    auto state_raw = std::make_shared<Tensor>();
    state_raw->Assign<float>({0.0f}, {1});

    feather::kernel::cuda_detail::ClearTensorCache();
    StaticGraph static_graph;
    static_graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("cache_state", cache_state), 0);
    ASSERT_EQ(static_graph.SetTensor("bias", bias), 0);
    ASSERT_EQ(static_graph.SetTensor("state_raw", state_raw), 0);
    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.ApplyPasses(), 0);
    ASSERT_EQ(static_graph.NodeSize(), 2U);
    EXPECT_EQ(static_graph.GetTensor("next_cache_state"), static_graph.GetTensor("state_raw"));

    RuntimeGraph runtime;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime), 0);
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(runtime.Run(), 0);
        ASSERT_EQ(feather::kernel::cuda_detail::SynchronizeInferenceStream(), 0);
    }

    const auto next_state = runtime.GetTensor("next_cache_state");
    ASSERT_NE(next_state, nullptr);
    ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHost(next_state.get(), sizeof(float), next_state->raw_data()),
              0);
    EXPECT_FLOAT_EQ(next_state->data<float>()[0], 3.0f);
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(runtime_graph_test, CudaViewAliasSharesDeviceAllocationAndLifetime) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1.0f, 2.0f}, {1, 2});
    auto output = std::make_shared<Tensor>();
    output->Assign<float>({0.0f, 0.0f}, {2, 1});

    void* input_device = nullptr;
    void* output_device = nullptr;
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      input.get(), 2 * sizeof(float), input->data<float>(), &input_device),
                  0);
        ASSERT_NE(input_device, nullptr);
        ASSERT_EQ(feather::kernel::cuda_detail::AliasTensorDeviceStorage(
                      input.get(), output.get(), 2 * sizeof(float)),
                  0);
        ASSERT_EQ(feather::kernel::cuda_detail::AcquireTensorDevice(
                      output.get(), 2 * sizeof(float), output->data<float>(), &output_device),
                  0);
        EXPECT_EQ(output_device, input_device);

        feather::kernel::cuda_detail::ReleaseTensorDevice(input.get());
        ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHost(
                      output.get(), 2 * sizeof(float), output->raw_data()),
                  0);
    }
    EXPECT_FLOAT_EQ(output->data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(output->data<float>()[1], 2.0f);
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}


TEST(runtime_graph_test, CudaGraphReleasesDeadTensorCachesIntoPool) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::kernel::cuda_detail::ClearTensorCache();

    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1.0f, 2.0f}, {2});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({10.0f, 20.0f}, {2});
    auto tmp0 = std::make_shared<Tensor>();
    tmp0->Assign<float>({0.0f, 0.0f}, {2});
    auto tmp1 = std::make_shared<Tensor>();
    tmp1->Assign<float>({0.0f, 0.0f}, {2});
    auto output = std::make_shared<Tensor>();
    output->Assign<float>({0.0f, 0.0f}, {2});

    feather::kernel::cuda_detail::MarkTensorDevicePersistent(bias.get(), true);

    feather::operators::BinaryParam add0_param{};
    add0_param.lhs = input;
    add0_param.rhs = bias;
    add0_param.out = tmp0;
    auto add0_op = std::make_shared<feather::operators::AddOp>("cuda_add0", add0_param);
    auto add0_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(add0_kernel, nullptr);
    add0_op->AttachKernel(std::move(add0_kernel));
    auto add0_detached = add0_op->DetachKernel();

    feather::operators::BinaryParam add1_param{};
    add1_param.lhs = tmp0;
    add1_param.rhs = bias;
    add1_param.out = tmp1;
    auto add1_op = std::make_shared<feather::operators::AddOp>("cuda_add1", add1_param);
    auto add1_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(add1_kernel, nullptr);
    add1_op->AttachKernel(std::move(add1_kernel));
    auto add1_detached = add1_op->DetachKernel();

    feather::operators::BinaryParam add2_param{};
    add2_param.lhs = tmp1;
    add2_param.rhs = bias;
    add2_param.out = output;
    auto add2_op = std::make_shared<feather::operators::AddOp>("cuda_add2", add2_param);
    auto add2_kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(add2_kernel, nullptr);
    add2_op->AttachKernel(std::move(add2_kernel));
    auto add2_detached = add2_op->DetachKernel();

    RuntimeGraph graph;
    graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    graph.SetOutputNames({"output"});
    ASSERT_EQ(graph.SetTensor("input", input), 0);
    ASSERT_EQ(graph.SetTensor("bias", bias), 0);
    ASSERT_EQ(graph.SetTensor("tmp0", tmp0), 0);
    ASSERT_EQ(graph.SetTensor("tmp1", tmp1), 0);
    ASSERT_EQ(graph.SetTensor("output", output), 0);

    graph.AddNode(MakeBinaryRuntimeNode("cuda_add0", "Add", add0_op, std::move(add0_detached), {"input", "bias"},
                                        {"tmp0"}));
    graph.AddNode(MakeBinaryRuntimeNode("cuda_add1", "Add", add1_op, std::move(add1_detached), {"tmp0", "bias"},
                                        {"tmp1"}));
    graph.AddNode(MakeBinaryRuntimeNode("cuda_add2", "Add", add2_op, std::move(add2_detached), {"tmp1", "bias"},
                                        {"output"}));
    ASSERT_EQ(graph.Finalize(), 0);

    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_sync;
        ASSERT_EQ(graph.Run(), 0);
        ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHost(output.get(), output->numel() * sizeof(float),
                                                                 output->mutable_data<float>()),
                  0);
    }

    const auto stats = feather::kernel::cuda_detail::GetTensorCacheStats();
    EXPECT_EQ(stats.active_tensor_count, 2U);
    EXPECT_GE(stats.free_block_count, 1U);
    EXPECT_EQ(stats.persistent_tensor_count, 1U);
    EXPECT_FLOAT_EQ(output->data<float>()[0], 31.0f);
    EXPECT_FLOAT_EQ(output->data<float>()[1], 62.0f);

    feather::kernel::cuda_detail::ReleaseTensorDevice(output.get());
    feather::kernel::cuda_detail::ReleaseTensorDevice(bias.get());
    feather::kernel::cuda_detail::ClearTensorCache();
#endif
}

TEST(static_graph_test, BuildFailsWhenRequiredTensorMissing) {
    ModelDesc model;
    model.name = "missing_weight";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2}, {1, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}

TEST(static_graph_test, BuildFailsWhenFcShapeIsInvalid) {
    ModelDesc model;
    model.name = "bad_fc_shape";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3}, {1, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("weight", weight_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}

TEST(runtime_graph_test, LoweringProducesExecutableRuntimeNode) {
    ModelDesc model;
    model.name = "fc_graph_bound";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {1, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {1, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "fc0";
    node.op_type = "FC";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2}, {1, 2});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 1U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_FLOAT_EQ(output_tensor->data<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(output_tensor->data<float>()[1], 10.0f);
}

TEST(runtime_graph_test, LoweringPreservesDagDependencies) {
    ModelDesc model;
    model.name = "branch_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"left_out", "right_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc add_bias;
    add_bias.tensor.name = "add_bias";
    add_bias.tensor.dims = {2, 2};
    add_bias.tensor.data_type = DataType::FP32;
    add_bias.constant = true;

    ValueDesc mid;
    mid.tensor.name = "mid";
    mid.tensor.dims = {2, 2};
    mid.tensor.data_type = DataType::FP32;

    ValueDesc left_bias;
    left_bias.tensor.name = "left_bias";
    left_bias.tensor.dims = {2, 2};
    left_bias.tensor.data_type = DataType::FP32;
    left_bias.constant = true;

    ValueDesc right_scale;
    right_scale.tensor.name = "right_scale";
    right_scale.tensor.dims = {2, 2};
    right_scale.tensor.data_type = DataType::FP32;
    right_scale.constant = true;

    ValueDesc left_out;
    left_out.tensor.name = "left_out";
    left_out.tensor.dims = {2, 2};
    left_out.tensor.data_type = DataType::FP32;

    ValueDesc right_out;
    right_out.tensor.name = "right_out";
    right_out.tensor.dims = {2, 2};
    right_out.tensor.data_type = DataType::FP32;

    NodeDesc add0;
    add0.name = "add0";
    add0.op_type = "Add";
    add0.inputs = {"input", "add_bias"};
    add0.outputs = {"mid"};

    NodeDesc add1;
    add1.name = "add1";
    add1.op_type = "Add";
    add1.inputs = {"mid", "left_bias"};
    add1.outputs = {"left_out"};

    NodeDesc mul0;
    mul0.name = "mul0";
    mul0.op_type = "Mul";
    mul0.inputs = {"mid", "right_scale"};
    mul0.outputs = {"right_out"};

    model.graph.values = {input, add_bias, mid, left_bias, right_scale, left_out, right_out};
    model.graph.nodes = {add0, add1, mul0};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto add_bias_tensor = std::make_shared<Tensor>();
    add_bias_tensor->Assign<float>({1, 1, 1, 1}, {2, 2});

    auto left_bias_tensor = std::make_shared<Tensor>();
    left_bias_tensor->Assign<float>({2, 2, 2, 2}, {2, 2});

    auto right_scale_tensor = std::make_shared<Tensor>();
    right_scale_tensor->Assign<float>({3, 3, 3, 3}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("add_bias", add_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("left_bias", left_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("right_scale", right_scale_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 3U);

    const auto* add_node = runtime_graph.GetNode("add0");
    const auto* left_node = runtime_graph.GetNode("add1");
    const auto* right_node = runtime_graph.GetNode("mul0");
    ASSERT_NE(add_node, nullptr);
    ASSERT_NE(left_node, nullptr);
    ASSERT_NE(right_node, nullptr);

    EXPECT_EQ(add_node->pending_dependencies, 0U);
    EXPECT_EQ(add_node->successors.size(), 2U);
    EXPECT_EQ(left_node->pending_dependencies, 1U);
    EXPECT_EQ(right_node->pending_dependencies, 1U);
}

TEST(runtime_graph_test, LoweringInitializesReusableExecutionWorkers) {
    ModelDesc model;
    model.name = "branch_graph_workers";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"left_out", "right_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc add_bias;
    add_bias.tensor.name = "add_bias";
    add_bias.tensor.dims = {2, 2};
    add_bias.tensor.data_type = DataType::FP32;
    add_bias.constant = true;

    ValueDesc mid;
    mid.tensor.name = "mid";
    mid.tensor.dims = {2, 2};
    mid.tensor.data_type = DataType::FP32;

    ValueDesc left_bias;
    left_bias.tensor.name = "left_bias";
    left_bias.tensor.dims = {2, 2};
    left_bias.tensor.data_type = DataType::FP32;
    left_bias.constant = true;

    ValueDesc right_scale;
    right_scale.tensor.name = "right_scale";
    right_scale.tensor.dims = {2, 2};
    right_scale.tensor.data_type = DataType::FP32;
    right_scale.constant = true;

    ValueDesc left_out;
    left_out.tensor.name = "left_out";
    left_out.tensor.dims = {2, 2};
    left_out.tensor.data_type = DataType::FP32;

    ValueDesc right_out;
    right_out.tensor.name = "right_out";
    right_out.tensor.dims = {2, 2};
    right_out.tensor.data_type = DataType::FP32;

    NodeDesc add0;
    add0.name = "add0";
    add0.op_type = "Add";
    add0.inputs = {"input", "add_bias"};
    add0.outputs = {"mid"};

    NodeDesc add1;
    add1.name = "add1";
    add1.op_type = "Add";
    add1.inputs = {"mid", "left_bias"};
    add1.outputs = {"left_out"};

    NodeDesc mul0;
    mul0.name = "mul0";
    mul0.op_type = "Mul";
    mul0.inputs = {"mid", "right_scale"};
    mul0.outputs = {"right_out"};

    model.graph.values = {input, add_bias, mid, left_bias, right_scale, left_out, right_out};
    model.graph.nodes = {add0, add1, mul0};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto add_bias_tensor = std::make_shared<Tensor>();
    add_bias_tensor->Assign<float>({1, 1, 1, 1}, {2, 2});

    auto left_bias_tensor = std::make_shared<Tensor>();
    left_bias_tensor->Assign<float>({2, 2, 2, 2}, {2, 2});

    auto right_scale_tensor = std::make_shared<Tensor>();
    right_scale_tensor->Assign<float>({3, 3, 3, 3}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("add_bias", add_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("left_bias", left_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("right_scale", right_scale_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);

    const size_t expected_workers = std::min(feather::DefaultThreadCount(), runtime_graph.NodeSize());
    EXPECT_EQ(runtime_graph.WorkerCount(), expected_workers);

    ASSERT_EQ(runtime_graph.Run(), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);
    EXPECT_EQ(runtime_graph.WorkerCount(), expected_workers);
}

TEST(runtime_graph_test, RuntimeNodeProfileLabelUsesNodeNameAndOpType) {
    feather::RuntimeNode node;
    node.name = "conv_17";
    node.op_type = "Conv2D";
    EXPECT_EQ(node.ProfileLabel(), "RuntimeNode::conv_17[Conv2D]");
}
