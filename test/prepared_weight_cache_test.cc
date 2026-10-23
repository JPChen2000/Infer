#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <vector>

#include "core/tensor.h"
#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp8.h"
#include "util/bf16.h"

#ifdef FEATHER_WITH_X86
namespace {
using namespace feather;
using namespace feather::kernel::x86;

Tensor ReadOnlyView(const std::shared_ptr<std::vector<uint16_t>>& bytes, DataType dtype = DataType::BF16) {
    Tensor tensor;
    tensor.Resize(std::vector<int64_t>{2, 2});
    tensor.set_data_type(dtype);
    auto buffer = std::make_shared<Buffer>(static_cast<const void*>(bytes->data()), bytes->size() * 2, bytes);
    tensor.ResetBuffer(buffer, bytes->size() * 2);
    tensor.set_immutable(true);
    return tensor;
}

template<class Packed>
bool Prepare(Packed& pack, const Tensor& tensor) { return pack.Pack(tensor, 2, 2); }

template<class Packed>
void CheckSharing(DataType dtype) {
    auto bytes = std::make_shared<std::vector<uint16_t>>(4, FloatToBFloat16(1));
    Tensor first = ReadOnlyView(bytes, dtype), second = ReadOnlyView(bytes, dtype);
    ASSERT_NE(first.storage()->id(), second.storage()->id());
    ASSERT_NE(first.mutation_version(), second.mutation_version());
    Packed a, b;
    ASSERT_TRUE(Prepare(a, first));
    ASSERT_TRUE(Prepare(b, second));
    EXPECT_EQ(a.data(), b.data());
}

TEST(PreparedWeightCache, SharesBf16AcrossIndependentReadonlyStorage) { CheckSharing<PackedBf16Rhs>(DataType::BF16); }
TEST(PreparedWeightCache, SharesTransposedBf16) { CheckSharing<PackedBf16TransposedRhs>(DataType::BF16); }
TEST(PreparedWeightCache, SharesFp8E4M3) { CheckSharing<PackedFp8Rhs>(DataType::FP8E4M3); }
TEST(PreparedWeightCache, SharesTransposedFp8E5M2) { CheckSharing<PackedFp8TransposedRhs>(DataType::FP8E5M2); }

TEST(PreparedWeightCache, CowRepackPreservesOtherRequestAndComputesWithPrivateWorkspace) {
    auto bytes = std::make_shared<std::vector<uint16_t>>(
        std::initializer_list<uint16_t>{FloatToBFloat16(1), FloatToBFloat16(2), FloatToBFloat16(3), FloatToBFloat16(4)});
    Tensor first = ReadOnlyView(bytes), second = ReadOnlyView(bytes);
    PackedBf16Rhs a, b;
    ASSERT_TRUE(Prepare(a, first));
    ASSERT_TRUE(Prepare(b, second));
    // Tracked write detaches readonly backing without changing BF16 metadata.
    static_cast<uint16_t*>(first.mutable_data(8))[0] = FloatToBFloat16(5);
    ASSERT_TRUE(Prepare(a, first));
    EXPECT_NE(a.data(), b.data());
    uint16_t lhs[] = {FloatToBFloat16(1), FloatToBFloat16(1)}, out_a[2], out_b[2];
    Bf16LinearWorkspace wa, wb;
    ASSERT_EQ(0, ComputeLinearRowMajorX86Bf16PackedRhs(lhs, first.data<uint16_t>(), a, nullptr, 1, 2, 2,
        LinearBiasType::kNone, out_a, &wa, first.mutation_version()));
    ASSERT_EQ(0, ComputeLinearRowMajorX86Bf16PackedRhs(lhs, second.data<uint16_t>(), b, nullptr, 1, 2, 2,
        LinearBiasType::kNone, out_b, &wb, second.mutation_version()));
    EXPECT_EQ(8, BFloat16ToFloat(out_a[0]));
    EXPECT_EQ(4, BFloat16ToFloat(out_b[0]));
    EXPECT_EQ(6, BFloat16ToFloat(out_b[1]));
}

TEST(PreparedWeightCache, ConcurrentPreparationSharesOneResultAndRetainsSourceOwner) {
    auto bytes = std::make_shared<std::vector<uint16_t>>(4, FloatToBFloat16(1));
    std::weak_ptr<std::vector<uint16_t>> lifetime = bytes;
    std::vector<PackedBf16Rhs> packs(8);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < packs.size(); ++i) workers.emplace_back([&, i] {
        Tensor view = ReadOnlyView(bytes);
        EXPECT_TRUE(Prepare(packs[i], view));
    });
    for (auto& worker : workers) worker.join();
    for (const auto& pack : packs) EXPECT_EQ(packs[0].data(), pack.data());
    bytes.reset();
    EXPECT_FALSE(lifetime.expired());
    packs.clear();
    EXPECT_TRUE(lifetime.expired());
}

TEST(PreparedWeightCache, SeparatesPackingFormatsDimensionsAndSourceViews) {
    auto bytes = std::make_shared<std::vector<uint16_t>>(
        std::initializer_list<uint16_t>{FloatToBFloat16(1), FloatToBFloat16(2), FloatToBFloat16(3), FloatToBFloat16(4)});
    Tensor tensor = ReadOnlyView(bytes);
    PackedBf16Rhs normal, reshaped, offset;
    PackedBf16TransposedRhs transposed;
    ASSERT_TRUE(Prepare(normal, tensor));
    ASSERT_TRUE(Prepare(transposed, tensor));
    ASSERT_TRUE(reshaped.Pack(tensor, 1, 4));
    EXPECT_NE(normal.data(), transposed.data());
    EXPECT_NE(normal.data(), reshaped.data());
    EXPECT_EQ(BFloat16ToFloat(normal.data()[1]), 2);
    EXPECT_EQ(BFloat16ToFloat(transposed.data()[1]), 3);
    Tensor view = ReadOnlyView(bytes);
    view.ResetStorage(tensor.storage(), 4, 4);
    view.Resize(std::vector<int64_t>{1, 2});
    ASSERT_TRUE(offset.Pack(view, 1, 2));
    EXPECT_NE(offset.data(), normal.data());
    EXPECT_EQ(BFloat16ToFloat(offset.data()[0]), 3);
    EXPECT_FALSE(offset.Pack(view, 2, 2)); // The view has only two elements.

    tensor.set_data_type(DataType::FP8E4M3);
    PackedFp8Rhs e4m3, e5m2;
    ASSERT_TRUE(Prepare(e4m3, tensor));
    tensor.set_data_type(DataType::FP8E5M2);
    ASSERT_TRUE(Prepare(e5m2, tensor));
    EXPECT_NE(e4m3.data(), e5m2.data());
}

TEST(PreparedWeightCache, MutableWeightsAreNotSharedEvenIfMarkedImmutable) {
    Tensor tensor;
    tensor.Resize(std::vector<int64_t>{2, 2});
    const uint16_t values[] = {1, 2, 3, 4};
    tensor.Assign(values, DDim(std::vector<int64_t>{2, 2}));
    tensor.set_data_type(DataType::BF16);
    tensor.set_immutable(true);
    PackedBf16Rhs a, b;
    ASSERT_TRUE(Prepare(a, tensor));
    ASSERT_TRUE(Prepare(b, tensor));
    EXPECT_NE(a.data(), b.data());
}
} // namespace
#endif
