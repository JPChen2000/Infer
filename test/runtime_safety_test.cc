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

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <limits>
#include <thread>

#include "core/graph_lowering.h"
#include "core/shape_utils.h"
#include "model/model_io.h"
#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace {

model::ValueDesc Value(const std::string& name) {
    model::ValueDesc v;
    v.tensor.name = name; v.tensor.dims = {1}; v.tensor.data_type = DataType::FP32;
    return v;
}
std::shared_ptr<Tensor> FloatTensor(float value) {
    auto t = std::make_shared<Tensor>(); t->Assign<float>({value}, {1}); return t;
}
model::ModelDesc ReluModel() {
    model::ModelDesc m;
    m.graph.inputs = {"x"}; m.graph.outputs = {"y"};
    m.graph.values = {Value("x"), Value("y")};
    model::NodeDesc n;
    n.name = "relu"; n.op_type = "ReLU"; n.inputs = {"x"}; n.outputs = {"y"};
    m.graph.nodes = {n}; return m;
}

TEST(runtime_safety_test, ResetAndClearPreserveLiveStorageAliases) {
    auto pool = std::make_shared<BufferPool>();
    auto buffer = pool->allocate(64);
    Tensor original(buffer); original.Resize({16}); original.mutable_data<float>()[0] = 11;
    Tensor alias; alias.ShareDataWith(original);
    const auto* old = alias.raw_data(); buffer.reset();
    original.ResetBuffer(std::make_shared<Buffer>(64), 64);
    auto fresh = pool->allocate(64);
    EXPECT_NE(fresh->data(), old);
    EXPECT_FLOAT_EQ(alias.data<float>()[0], 11);
    Tensor second_alias; second_alias.ShareDataWith(alias);
    alias.clear();
    EXPECT_FALSE(alias.IsInitialized());
    EXPECT_FLOAT_EQ(second_alias.data<float>()[0], 11);
    second_alias.clear();
    auto reused = pool->allocate(64);
    EXPECT_EQ(reused->data(), old);
}

TEST(runtime_safety_test, PoolReleaseDropsOnlyTheCallingReference) {
    auto pool = std::make_shared<BufferPool>();
    auto first = pool->allocate(64); auto alias = first;
    pool->deallocate(first);
    EXPECT_EQ(first, nullptr);
    auto second = pool->allocate(64);
    EXPECT_NE(second->data(), alias->data());
    EXPECT_THROW(pool->deallocate_all(), std::logic_error);
}

TEST(runtime_safety_test, OffsetBoundsAndReallocationAreChecked) {
    Tensor t; t.Resize({1});
    auto storage = std::make_shared<Buffer>(64);
    EXPECT_THROW(t.ResetBuffer(storage, 8, 60), std::logic_error);
    EXPECT_THROW(t.ResetBuffer(storage, 0, 65), std::logic_error);
    t.ResetBuffer(storage, 4, 60);
    t.mutable_data<float>()[0] = 7;
    t.Resize({2});
    EXPECT_THROW(t.mutable_data<float>(), std::logic_error);
    EXPECT_THROW(t.data<float>(), std::logic_error);
    EXPECT_THROW(t.mutable_data(8), std::logic_error);
    t.Assign<float>({1, 2}, {2});
    EXPECT_EQ(t.offset(), 0u);
    EXPECT_FLOAT_EQ(t.data<float>()[1], 2);
    size_t bytes = 0;
    EXPECT_FALSE(CheckedTensorBytes({-1}, DataType::FP32, &bytes));
    EXPECT_FALSE(CheckedTensorBytes({std::numeric_limits<int64_t>::max(), 2}, DataType::FP32, &bytes));
    EXPECT_TRUE(CheckedTensorBytes({3}, DataType::INT4, &bytes));
    EXPECT_EQ(bytes, 2u);
}

TEST(runtime_safety_test, WeightViewsValidateShapeAndCopyBeforeWriting) {
    const std::string path = "/tmp/feather_safety_weight.bin";
    const float weights[] = {11, 22, 33, 44};
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(weights), sizeof(weights)); }
    model::WeightStore store;
    model::TensorDesc desc; desc.dims = {1024}; desc.data_type = DataType::FP32;
    model::WeightLocation loc; loc.shard_path = path; loc.byte_size = 4;
    EXPECT_EQ(store.CreateTensorView(desc, loc), nullptr);
    desc.dims = {-1}; EXPECT_EQ(store.CreateTensorView(desc, loc), nullptr);
    desc.dims = {1}; loc.offset = 4;
    auto first = store.CreateTensorView(desc, loc);
    auto second = store.CreateTensorView(desc, loc);
    ASSERT_NE(first, nullptr); ASSERT_NE(second, nullptr);
    EXPECT_FLOAT_EQ(first->data<float>()[0], 22);
    first->mutable_data<float>()[0] = 99;
    EXPECT_FLOAT_EQ(second->data<float>()[0], 22);
    first->Resize({2}); EXPECT_THROW(first->data<float>(), std::logic_error);
}

TEST(runtime_safety_test, MalformedModelSizesFailWithoutThrowingOrReplacingLoadedModel) {
    const std::string path = "/tmp/feather_safety_bad.fth";
    model::ModelWriter writer; auto m = ReluModel(); m.name = "valid";
    ASSERT_TRUE(writer.Save(path, m, {}));
    model::ModelLoader loader; ASSERT_TRUE(loader.Load(path));
    struct Header { char magic[8]; uint32_t version, reserved; uint64_t metadata; };
    Header h{{'F','T','H','M','O','D','L','\0'}, 3, 0, std::numeric_limits<uint64_t>::max()};
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<char*>(&h), sizeof(h)); }
    EXPECT_NO_THROW(EXPECT_FALSE(loader.Load(path)));
    EXPECT_EQ(loader.model().name, "valid");
    // Small metadata with an enormous vector count must be rejected before resize.
    h.metadata = 32;
    { std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<char*>(&h), sizeof(h));
      const uint64_t data[] = {0, 1, 0, std::numeric_limits<uint64_t>::max()};
      out.write(reinterpret_cast<const char*>(data), sizeof(data)); }
    EXPECT_NO_THROW(EXPECT_FALSE(loader.Load(path)));
}

TEST(runtime_safety_test, RebindingInputUpdatesCompiledKernelsAndRejectsIncompatibleInputs) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(ReluModel()), 0); graph.SetTensor("x", FloatTensor(1));
    ASSERT_EQ(graph.Build(), 0);
    RuntimeGraph runtime; GraphLowering lower; ASSERT_EQ(lower.Lower(graph, &runtime), 0);
    const auto identity = runtime.GetTensor("x");
    auto replacement = FloatTensor(9); replacement->set_immutable(true);
    ASSERT_EQ(runtime.SetTensor("x", replacement), 0);
    EXPECT_EQ(runtime.GetTensor("x"), identity);
    ASSERT_EQ(runtime.Run(), 0);
    EXPECT_FLOAT_EQ(runtime.GetTensor("y")->data<float>()[0], 9);
    EXPECT_FALSE(identity->is_immutable());
    ASSERT_EQ(runtime.SetTensor("x", FloatTensor(8)), 0);
    ASSERT_EQ(runtime.Run(), 0);
    EXPECT_FLOAT_EQ(runtime.GetTensor("y")->data<float>()[0], 8);
    auto wrong_shape = std::make_shared<Tensor>(); wrong_shape->Assign<float>({1, 2}, {2});
    EXPECT_EQ(runtime.SetTensor("x", wrong_shape), -1);
    EXPECT_EQ(runtime.SetTensor("y", FloatTensor(3)), -1);
    EXPECT_EQ(runtime.SetTensor("unknown", FloatTensor(3)), -1);
}

TEST(runtime_safety_test, RejectsMissingSourcesDuplicateNamesAndCycles) {
    const auto rejects = [](model::ModelDesc model) {
        StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
        graph.SetModel(model); graph.SetTensor("x", FloatTensor(1));
        EXPECT_FALSE(graph.BuildStatus().ok());
        RuntimeGraph runtime; GraphLowering lower;
        EXPECT_FALSE(lower.LowerStatus(graph, &runtime).ok());
    };
    auto m = ReluModel(); m.graph.values.push_back(Value("orphan"));
    m.graph.nodes[0].inputs = {"orphan"}; rejects(m);
    m = ReluModel(); m.graph.nodes.push_back(m.graph.nodes[0]); rejects(m);
    m = ReluModel(); m.graph.values.push_back(m.graph.values[0]); rejects(m);
    m = ReluModel(); m.graph.outputs = {"unknown"}; rejects(m);
    m = ReluModel(); m.graph.nodes[0].inputs = {"y"}; rejects(m);
    RuntimeGraph runtime; RuntimeNode a, b;
    a.name = "a"; a.inputs = {"b_out"}; a.outputs = {"a_out"};
    b.name = "b"; b.inputs = {"a_out"}; b.outputs = {"b_out"};
    runtime.AddNode(std::move(a)); runtime.AddNode(std::move(b));
    EXPECT_FALSE(runtime.FinalizeStatus().ok());
}

TEST(runtime_safety_test, BuildsUnorderedDagAndAssignsNamesToAnonymousNodes) {
    auto m = ReluModel(); m.graph.values.push_back(Value("middle"));
    auto first = m.graph.nodes[0]; first.name.clear(); first.outputs = {"middle"};
    m.graph.nodes[0].inputs = {"middle"};
    m.graph.nodes.push_back(first); // Producer deliberately follows consumer.
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    graph.SetModel(m); graph.SetTensor("x", FloatTensor(7));
    ASSERT_EQ(graph.Build(), 0);
    RuntimeGraph runtime; GraphLowering lower; ASSERT_EQ(lower.Lower(graph, &runtime), 0);
    ASSERT_EQ(runtime.Run(), 0); EXPECT_FLOAT_EQ(runtime.GetTensor("y")->data<float>()[0], 7);
}

TEST(runtime_safety_test, ExplicitX86RequestFallsBackWhenNativeBackendIsDisabled) {
    const char* old = std::getenv("FEATHER_DISABLE_X86");
    const bool had_value = old != nullptr; const std::string previous = old == nullptr ? "" : old;
    setenv("FEATHER_DISABLE_X86", "1", 1);
    EXPECT_EQ(GetHostRuntimeDevice(), DeviceType::COMMON);
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "ReLU");
    EXPECT_NE(kernel, nullptr);
    if (kernel != nullptr) EXPECT_EQ(kernel->device(), DeviceType::COMMON);
    if (had_value) setenv("FEATHER_DISABLE_X86", previous.c_str(), 1); else unsetenv("FEATHER_DISABLE_X86");
}

#ifdef FEATHER_WITH_CUDA
namespace cd = kernel::cuda_detail;
TEST(runtime_safety_test, CudaContextsIsolateFlagsAcrossSessionsAndThreads) {
    const auto first = cd::CreateExecutionContext(); const auto second = cd::CreateExecutionContext();
    { cd::ExecutionContextScope scope(first); cd::DeferredHostSyncScope deferred;
      EXPECT_TRUE(cd::DeferredHostSyncEnabled());
      std::thread other([&] { cd::ExecutionContextScope scope2(second);
          EXPECT_FALSE(cd::DeferredHostSyncEnabled());
          cd::DeferredHostSyncScope nested; EXPECT_TRUE(cd::DeferredHostSyncEnabled()); });
      other.join(); EXPECT_TRUE(cd::DeferredHostSyncEnabled()); }
    { cd::ExecutionContextScope scope(first); EXPECT_FALSE(cd::DeferredHostSyncEnabled()); }
    { cd::ExecutionContextScope scope(second); EXPECT_FALSE(cd::DeferredHostSyncEnabled()); }
}
TEST(runtime_safety_test, DeviceReadsDoNotMarkStaleHostCopiesValidAndCachesAreIsolated) {
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    const auto first = cd::CreateExecutionContext(); const auto second = cd::CreateExecutionContext();
    auto t = FloatTensor(-1); void* device = nullptr;
    { cd::ExecutionContextScope scope(first); cd::DeferredHostSyncScope deferred;
      ASSERT_EQ(cd::AcquireOutputTensorDevice(t.get(), sizeof(float), &device), 0);
      const float produced = 42;
      ASSERT_EQ(cudaMemcpyAsync(device, &produced, sizeof(float), cudaMemcpyHostToDevice, cd::InferenceStream()), cudaSuccess);
      void* read_ptr = nullptr;
      ASSERT_EQ(cd::AcquireTensorDevice(t.get(), sizeof(float), t->raw_data(), &read_ptr), 0);
      { cd::ExecutionContextScope other(second); cd::ClearTensorCache(); EXPECT_EQ(cd::GetTensorCacheStats().active_bytes, 0u); }
      ASSERT_EQ(cd::SyncTensorToHostIfNeeded(t.get(), sizeof(float), t->raw_data()), 0);
      EXPECT_FLOAT_EQ(t->data<float>()[0], 42);
      EXPECT_GT(cd::GetTensorCacheStats().active_bytes, 0u); }
}
TEST(runtime_safety_test, DeviceDownloadCopiesReadonlyWeightsBeforeTrackedWrite) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    auto owner = FloatTensor(2);
    auto readonly = std::make_shared<Buffer>(owner->mutable_data(sizeof(float)), sizeof(float), owner, true);
    Tensor target, other;
    target.Resize({1}); target.set_data_type(DataType::FP32);
    other.Resize({1}); other.set_data_type(DataType::FP32);
    target.ResetBuffer(readonly, sizeof(float)); other.ResetBuffer(readonly, sizeof(float));
    Tensor alias; alias.ShareDataWith(target);
    const auto old_version = target.mutation_version();
    const auto other_version = other.mutation_version();
    const auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    cd::DeferredHostSyncScope deferred;
    void* device = nullptr;
    ASSERT_EQ(cd::AcquireOutputTensorDevice(&target, sizeof(float), &device), 0);
    EXPECT_EQ(target.mutation_version(), old_version);
    const float produced = 42;
    ASSERT_EQ(cudaMemcpyAsync(device, &produced, sizeof(float), cudaMemcpyHostToDevice, cd::InferenceStream()),
              cudaSuccess);
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&target, 0, nullptr), 0);
    EXPECT_EQ(target.mutation_version(), old_version);
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&target, sizeof(float), target.raw_data()), 0);
    EXPECT_FLOAT_EQ(target.data<float>()[0], 42);
    EXPECT_FLOAT_EQ(alias.data<float>()[0], 42);
    EXPECT_NE(target.mutation_version(), old_version);
    EXPECT_EQ(alias.mutation_version(), target.mutation_version());
    EXPECT_FLOAT_EQ(other.data<float>()[0], 2);
    EXPECT_EQ(other.mutation_version(), other_version);
    const auto synced_version = target.mutation_version();
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&target, sizeof(float), target.raw_data()), 0);
    EXPECT_EQ(target.mutation_version(), synced_version);
}
TEST(runtime_safety_test, DownloadPreservesPendingReadonlyAliasViewsAndDetectsHostWrites) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    auto owner = std::make_shared<Tensor>(); owner->Assign<float>({1, 2}, {2});
    auto readonly = std::make_shared<Buffer>(owner->mutable_data(2 * sizeof(float)), 2 * sizeof(float), owner, true);
    Tensor first, second;
    first.Resize({1}); first.set_data_type(DataType::FP32); first.set_immutable(true);
    first.ResetBuffer(readonly, sizeof(float));
    second.Resize({1}); second.set_data_type(DataType::FP32); second.set_immutable(true);
    second.ResetStorage(first.storage(), sizeof(float), sizeof(float));
    const auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    cd::DeferredHostSyncScope deferred;
    void* first_device = nullptr;
    void* second_device = nullptr;
    ASSERT_EQ(cd::AcquireOutputTensorDevice(&first, sizeof(float), &first_device), 0);
    ASSERT_EQ(cd::AcquireOutputTensorDevice(&second, sizeof(float), &second_device), 0);
    const float first_result = 42, second_result = 99;
    ASSERT_EQ(cudaMemcpyAsync(first_device, &first_result, sizeof(float), cudaMemcpyHostToDevice, cd::InferenceStream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(second_device, &second_result, sizeof(float), cudaMemcpyHostToDevice, cd::InferenceStream()),
              cudaSuccess);
    const void* old_second_host = second.raw_data();
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&first, sizeof(float), first.raw_data()), 0);
    EXPECT_FLOAT_EQ(first.data<float>()[0], 42);
    EXPECT_FLOAT_EQ(second.data<float>()[0], 2);
    EXPECT_NE(second.raw_data(), old_second_host);
    void* read_device = nullptr;
    ASSERT_EQ(cd::AcquireTensorDevice(&second, sizeof(float), second.raw_data(), &read_device), 0);
    float observed = 0;
    ASSERT_EQ(cudaMemcpyAsync(&observed, read_device, sizeof(float), cudaMemcpyDeviceToHost, cd::InferenceStream()),
              cudaSuccess);
    ASSERT_EQ(cd::SynchronizeInferenceStream(), 0);
    EXPECT_FLOAT_EQ(observed, 99);
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&second, sizeof(float), second.raw_data()), 0);
    EXPECT_FLOAT_EQ(second.data<float>()[0], 99);
    EXPECT_FLOAT_EQ(owner->data<float>()[1], 2);

    // An actual host write still supersedes a pending device value.
    ASSERT_EQ(cd::AcquireOutputTensorDevice(&second, sizeof(float), &second_device), 0);
    second.mutable_data<float>()[0] = 7;
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(&second, sizeof(float), second.raw_data()), 0);
    EXPECT_FLOAT_EQ(second.data<float>()[0], 7);
    ASSERT_EQ(cd::AcquireTensorDevice(&second, sizeof(float), second.raw_data(), &read_device), 0);
    ASSERT_EQ(cudaMemcpyAsync(&observed, read_device, sizeof(float), cudaMemcpyDeviceToHost, cd::InferenceStream()),
              cudaSuccess);
    ASSERT_EQ(cd::SynchronizeInferenceStream(), 0);
    EXPECT_FLOAT_EQ(observed, 7);
}
TEST(runtime_safety_test, DownloadInvalidatesIndependentDeviceCachesForOverlappingViews) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    auto producer = FloatTensor(1);
    Tensor alias; alias.ShareDataWith(*producer);
    const auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    cd::DeferredHostSyncScope deferred;
    void* alias_device = nullptr;
    ASSERT_EQ(cd::AcquireTensorDevice(&alias, sizeof(float), alias.raw_data(), &alias_device), 0);
    for (const bool pending_alias : {false, true}) {
        if (pending_alias) {
            ASSERT_EQ(cd::AcquireOutputTensorDevice(&alias, sizeof(float), &alias_device), 0);
            const float superseded = 77;
            ASSERT_EQ(cudaMemcpyAsync(alias_device, &superseded, sizeof(float), cudaMemcpyHostToDevice,
                                      cd::InferenceStream()), cudaSuccess);
            ASSERT_EQ(cd::SynchronizeInferenceStream(), 0);
        }
        void* produced_device = nullptr;
        ASSERT_EQ(cd::AcquireOutputTensorDevice(producer.get(), sizeof(float), &produced_device), 0);
        const float result = pending_alias ? 43 : 42;
        ASSERT_EQ(cudaMemcpyAsync(produced_device, &result, sizeof(float), cudaMemcpyHostToDevice,
                                  cd::InferenceStream()), cudaSuccess);
        ASSERT_EQ(cd::SyncTensorToHostIfNeeded(producer.get(), sizeof(float), producer->raw_data()), 0);
        ASSERT_EQ(cd::AcquireTensorDevice(&alias, sizeof(float), alias.raw_data(), &alias_device), 0);
        float observed = 0;
        ASSERT_EQ(cudaMemcpyAsync(&observed, alias_device, sizeof(float), cudaMemcpyDeviceToHost,
                                  cd::InferenceStream()), cudaSuccess);
        ASSERT_EQ(cd::SynchronizeInferenceStream(), 0);
        EXPECT_FLOAT_EQ(observed, result);
    }
}
TEST(runtime_safety_test, PartialDownloadRejectsOverwritingBroaderPendingDeviceView) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    auto owner = std::make_shared<Tensor>(); owner->Assign<float>({1, 2}, {2});
    auto readonly = std::make_shared<Buffer>(owner->mutable_data(2 * sizeof(float)), 2 * sizeof(float), owner, true);
    auto broad = std::make_shared<Tensor>(); broad->Resize({2}); broad->set_data_type(DataType::FP32);
    broad->ResetBuffer(readonly, 2 * sizeof(float));
    Tensor narrow; narrow.Resize({1}); narrow.set_data_type(DataType::FP32);
    narrow.ResetStorage(broad->storage(), sizeof(float));
    const auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    cd::DeferredHostSyncScope deferred;
    void* broad_device = nullptr;
    void* narrow_device = nullptr;
    ASSERT_EQ(cd::AcquireOutputTensorDevice(broad.get(), 2 * sizeof(float), &broad_device), 0);
    ASSERT_EQ(cd::AcquireOutputTensorDevice(&narrow, sizeof(float), &narrow_device), 0);
    const float broad_result[] = {77, 88};
    const float narrow_result = 42;
    ASSERT_EQ(cudaMemcpyAsync(broad_device, broad_result, sizeof(broad_result), cudaMemcpyHostToDevice,
                              cd::InferenceStream()), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(narrow_device, &narrow_result, sizeof(float), cudaMemcpyHostToDevice,
                              cd::InferenceStream()), cudaSuccess);
    const auto version = broad->mutation_version();
    const void* host_pointer = broad->raw_data();
    EXPECT_NE(cd::SyncTensorToHostIfNeeded(&narrow, sizeof(float), narrow.raw_data()), 0);
    EXPECT_NE(cd::SyncTensorToHost(&narrow, sizeof(float), narrow.raw_data()), 0);
    EXPECT_NE(cd::SyncTensorToHostIfNeeded(broad.get(), sizeof(float), broad->raw_data()), 0);
    EXPECT_NE(cd::SyncTensorToHost(broad.get(), sizeof(float), broad->raw_data()), 0);
    EXPECT_EQ(broad->mutation_version(), version);
    EXPECT_EQ(broad->raw_data(), host_pointer);
    EXPECT_TRUE(broad->storage()->read_only());
    EXPECT_FLOAT_EQ(broad->data<float>()[0], 1);
    EXPECT_FLOAT_EQ(broad->data<float>()[1], 2);
    ASSERT_EQ(cd::SyncTensorToHostIfNeeded(broad.get(), sizeof(broad_result), broad->raw_data()), 0);
    EXPECT_FLOAT_EQ(broad->data<float>()[0], 77);
    EXPECT_FLOAT_EQ(broad->data<float>()[1], 88);
}
#endif
}  // namespace
}  // namespace feather
