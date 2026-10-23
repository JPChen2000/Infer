#include <gtest/gtest.h>

#include <limits>
#include "core/device_replica.h"
#include "core/tensor.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace {
using detail::DeviceViewIdentity;
using Index = detail::DeviceReplicaIndex<int>;

TEST(ReplicaIdentity, EqualViewsShareStateAcrossHandles) {
    Index index;
    int first, alias;
    const DeviceViewIdentity view{17, 8, 16};
    index.Bind(&first, view, [](int&) {}) = 42;
    EXPECT_EQ(index.Bind(&alias, view, [](int&) {}), 42);
    EXPECT_EQ(index.size(), 1u);
    size_t released = 0;
    index.EraseHandle(&first, [&](int&) { ++released; });
    EXPECT_EQ(released, 0u);
    ASSERT_NE(index.FindHandle(&alias), index.end());
    EXPECT_EQ(index.FindHandle(&alias)->second, 42);
    index.EraseHandle(&alias, [&](int&) { ++released; });
    EXPECT_EQ(released, 1u);
    EXPECT_EQ(index.size(), 0u);
    EXPECT_TRUE(index.StorageViews(17).empty());
}

TEST(ReplicaIdentity, IdentityIncludesStorageOffsetAndByteRange) {
    Index index;
    int a, b, c, d;
    index.Bind(&a, {1, 0, 8}, [](int&) {}) = 1;
    index.Bind(&b, {2, 0, 8}, [](int&) {}) = 2;
    index.Bind(&c, {1, 4, 8}, [](int&) {}) = 3;
    index.Bind(&d, {1, 0, 4}, [](int&) {}) = 4;
    EXPECT_EQ(index.size(), 4u);
    EXPECT_EQ(index.StorageViews(1).size(), 3u);
    EXPECT_EQ(index.StorageViews(2).size(), 1u);
    EXPECT_TRUE(index.StorageViews(3).empty());
}

TEST(ReplicaIdentity, RebindingHandleRetiresOnlyUnreferencedOldView) {
    Index index;
    int handle, alias;
    index.Bind(&handle, {1, 0, 4}, [](int&) {}) = 7;
    index.Bind(&alias, {1, 0, 4}, [](int&) {});
    size_t released = 0;
    index.Bind(&handle, {2, 0, 4}, [&](int&) { ++released; }) = 9;
    EXPECT_EQ(released, 0u);
    EXPECT_EQ(index.StorageViews(1).size(), 1u);
    index.EraseHandle(&alias, [&](int& value) { EXPECT_EQ(value, 7); ++released; });
    EXPECT_EQ(released, 1u);
    EXPECT_TRUE(index.StorageViews(1).empty());
    EXPECT_EQ(index.FindHandle(&handle)->second, 9);
}

TEST(ReplicaIdentity, StorageLocalVisitsExcludeUnrelatedReplicas) {
    Index index;
    int handles[128];
    for (size_t i = 0; i < 128; ++i)
        index.Bind(&handles[i], {i + 1, 0, 4}, [](int&) {}) = static_cast<int>(i);
    int alias;
    index.Bind(&alias, {64, 4, 4}, [](int&) {}) = 999;
    size_t visited = 0;
    for (const auto& view : index.StorageViews(64)) {
        auto found = index.find(view);
        ASSERT_NE(found, index.end());
        EXPECT_TRUE(found->second == 63 || found->second == 999);
        ++visited;
    }
    EXPECT_EQ(visited, 2u);
}

TEST(ReplicaIdentity, SnapshotRestoresRegistrationsAndStorageIndex) {
    Index index;
    int handle;
    index.Bind(&handle, {1, 0, 8}, [](int&) {}) = 11;
    const auto snapshot = index;
    index.Bind(&handle, {2, 4, 8}, [](int&) {}) = 22;
    index = snapshot;
    EXPECT_EQ(index.FindHandle(&handle)->second, 11);
    EXPECT_EQ(index.StorageViews(1).size(), 1u);
    EXPECT_TRUE(index.StorageViews(2).empty());
}

TEST(ReplicaIdentity, ByteRangeRelationsAvoidOverflowAndRespectStorage) {
    const auto maximum = std::numeric_limits<size_t>::max();
    const DeviceViewIdentity whole{1, maximum - 8, 8};
    EXPECT_TRUE(whole.Contains({1, maximum - 4, 4}));
    EXPECT_TRUE(whole.Overlaps({1, maximum - 4, 4}));
    EXPECT_FALSE(whole.Contains({1, maximum - 4, 8}));
    EXPECT_FALSE(whole.Overlaps({2, maximum - 4, 4}));
    EXPECT_FALSE(whole.Overlaps({1, maximum - 8, 0}));
    EXPECT_FALSE((DeviceViewIdentity{1, 0, 4}.Overlaps({1, 4, 4})));
}

TEST(ReplicaIdentity, StorageMutationKeepsIdentityAndDistinctStorageSeparatesViews) {
    Tensor tensor;
    tensor.Assign<float>({1, 2}, {2});
    const auto id = tensor.storage()->id();
    const auto version = tensor.mutation_version();
    tensor.mutable_data<float>()[0] = 3;
    EXPECT_EQ(tensor.storage()->id(), id);
    EXPECT_NE(tensor.mutation_version(), version);
    Tensor alias;
    alias.ShareDataWith(tensor);
    EXPECT_EQ(alias.storage()->id(), id);
    Tensor distinct;
    distinct.Assign<float>({1, 2}, {2});
    EXPECT_NE(distinct.storage()->id(), id);
}

TEST(ReplicaIdentity, ReadOnlyCowKeepsAliasGroupIdentityAfterPointerMoves) {
    auto owner = std::make_shared<Tensor>();
    owner->Assign<float>({1, 2}, {2});
    auto readonly = std::make_shared<Buffer>(owner->raw_data(), 8, owner, true);
    Tensor tensor;
    tensor.Resize({2});
    tensor.set_data_type(DataType::FP32);
    tensor.ResetBuffer(readonly, 8);
    Tensor alias;
    alias.ShareDataWith(tensor);
    const auto id = tensor.storage()->id();
    const auto* pointer = tensor.raw_data();
    tensor.mutable_data<float>()[0] = 3;
    EXPECT_NE(tensor.raw_data(), pointer);
    EXPECT_EQ(tensor.storage()->id(), id);
    EXPECT_EQ(alias.storage()->id(), id);
    EXPECT_EQ(alias.data<float>()[0], 3);
    EXPECT_EQ(owner->data<float>()[0], 1);
}

#ifdef FEATHER_WITH_CUDA
// These exercise context/registration bookkeeping without creating a CUDA
// stream, device allocation, or hardware dependency.
TEST(ReplicaIdentity, ContextRegistrationReleasesSharedViewAfterFinalUser) {
    namespace cd = kernel::cuda_detail;
    auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    Tensor tensor;
    tensor.Assign<float>({1, 2}, {2});
    Tensor alias;
    alias.ShareDataWith(tensor);
    cd::RegisterTensorDeviceUser(&tensor);
    cd::RegisterTensorDeviceUser(&tensor);
    cd::RegisterTensorDeviceUser(&alias);
    cd::MarkTensorDevicePersistent(&tensor, true);
    cd::MarkTensorDevicePersistent(&alias, true);
    EXPECT_EQ(cd::GetTensorCacheStats().persistent_tensor_count, 1u);
    cd::UnregisterTensorDeviceUsers({&tensor});
    EXPECT_TRUE(cd::IsTensorDevicePersistent(&tensor));
    cd::UnregisterTensorDeviceUsers({&tensor});
    EXPECT_FALSE(cd::IsTensorDevicePersistent(&tensor));
    EXPECT_EQ(cd::GetTensorCacheStats().persistent_tensor_count, 1u);
    cd::UnregisterTensorDeviceUsers({&alias});
    EXPECT_EQ(cd::GetTensorCacheStats().persistent_tensor_count, 0u);
}

TEST(ReplicaIdentity, AbandonedCaptureRestoresViewAndHandleRegistrations) {
    namespace cd = kernel::cuda_detail;
    auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    Tensor tensor, temporary;
    tensor.Assign<float>({1}, {1});
    temporary.Assign<float>({2}, {1});
    cd::MarkTensorDevicePersistent(&tensor, true);
    {
        cd::CaptureAllocationScope capture;
        ASSERT_TRUE(capture.valid());
        cd::MarkTensorDevicePersistent(&temporary, true);
        EXPECT_EQ(cd::GetTensorCacheStats().persistent_tensor_count, 2u);
    }
    EXPECT_EQ(cd::GetTensorCacheStats().persistent_tensor_count, 1u);
    EXPECT_TRUE(cd::IsTensorDevicePersistent(&tensor));
    EXPECT_FALSE(cd::IsTensorDevicePersistent(&temporary));
}

TEST(ReplicaIdentity, CudaEqualViewsShareAllocationUntilFinalHandleRelease) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    namespace cd = kernel::cuda_detail;
    auto context = cd::CreateExecutionContext();
    cd::ExecutionContextScope scope(context);
    cd::DeferredHostSyncScope deferred;
    Tensor tensor;
    tensor.Assign<float>({1, 2}, {2});
    Tensor alias;
    alias.ShareDataWith(tensor);
    void* first = nullptr;
    void* second = nullptr;
    ASSERT_EQ(cd::AcquireTensorDevice(&tensor, 8, tensor.raw_data(), &first), 0);
    ASSERT_EQ(cd::AcquireTensorDevice(&alias, 8, alias.raw_data(), &second), 0);
    EXPECT_EQ(first, second);
    EXPECT_EQ(cd::GetTensorCacheStats().active_tensor_count, 1u);
    cd::ReleaseTensorDevice(&tensor);
    EXPECT_EQ(cd::GetTensorCacheStats().active_tensor_count, 1u);
    ASSERT_EQ(cd::AcquireTensorDevice(&alias, 8, alias.raw_data(), &second), 0);
    EXPECT_EQ(first, second);
    cd::ReleaseTensorDevice(&alias);
    EXPECT_EQ(cd::GetTensorCacheStats().active_tensor_count, 0u);
}
#endif
}  // namespace
}  // namespace feather
