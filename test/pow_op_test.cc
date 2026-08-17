#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/kernel.h"
#include "core/operator.h"
#include "core/operator_registry.h"
#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"
#include "src/operator/params.h"
#include "src/operator/pow_op.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::PowParam;

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

TEST(pow_op_test, PowRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f)},
                            {4});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{4});
    out->mutable_data<uint16_t>();

    PowParam param{};
    param.input = input;
    param.out = out;
    param.exponent = 2.0f;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::PowOp>("pow_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Pow");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(pow_op_test, UsesScalarExponentTensorAfterGraphLowering) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({-2.0f, -1.5f, 0.5f, 3.0f}, {4});
    auto exponent = std::make_shared<Tensor>();
    exponent->Assign<float>({2.0f}, {});

    feather::model::ModelDesc model;
    model.name = "pow_with_tensor_exponent";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"out"};
    model.graph.values = {
        feather::model::ValueDesc{{"input", {4}, DataType::FP32, feather::DataLayout::ND}, false},
        feather::model::ValueDesc{{"exponent", {}, DataType::FP32, feather::DataLayout::ND}, true},
        feather::model::ValueDesc{{"out", {4}, DataType::FP32, feather::DataLayout::ND}, false},
    };
    model.graph.nodes = {{"pow_with_tensor_exponent", "Pow", "", {"input", "exponent"}, {"out"}, {}}};

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input), 0);
    ASSERT_EQ(static_graph.SetTensor("exponent", exponent), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    const auto out = runtime_graph.GetTensor("out");
    ASSERT_NE(out, nullptr);
    const std::vector<float> expected = {4.0f, 2.25f, 0.25f, 9.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

#ifdef FEATHER_WITH_CUDA
TEST(pow_op_test, UsesScalarExponentTensorAfterCudaGraphLowering) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<Tensor>();
    input->Assign<float>({-2.0f, -1.5f, 0.5f, 3.0f}, {4});
    auto exponent = std::make_shared<Tensor>();
    exponent->Assign<float>({2.0f}, {});

    feather::model::ModelDesc model;
    model.name = "cuda_pow_with_tensor_exponent";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"out"};
    model.graph.values = {
        feather::model::ValueDesc{{"input", {4}, DataType::FP32, feather::DataLayout::ND}, false},
        feather::model::ValueDesc{{"exponent", {}, DataType::FP32, feather::DataLayout::ND}, true},
        feather::model::ValueDesc{{"out", {4}, DataType::FP32, feather::DataLayout::ND}, false},
    };
    model.graph.nodes = {{"cuda_pow_with_tensor_exponent", "Pow", "", {"input", "exponent"}, {"out"}, {}}};

    feather::StaticGraph static_graph;
    static_graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input), 0);
    ASSERT_EQ(static_graph.SetTensor("exponent", exponent), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    const auto* runtime_node = runtime_graph.GetNode("cuda_pow_with_tensor_exponent");
    ASSERT_NE(runtime_node, nullptr);
    ASSERT_EQ(runtime_node->kernel_device, DeviceType::CUDA);
    ASSERT_EQ(runtime_graph.Run(), 0);

    const auto out = runtime_graph.GetTensor("out");
    ASSERT_NE(out, nullptr);
    const std::vector<float> expected = {4.0f, 2.25f, 0.25f, 9.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}
#endif
