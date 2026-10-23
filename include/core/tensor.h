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

#ifndef FEATHER_INFER_DATA_TENSOR_H
#define FEATHER_INFER_DATA_TENSOR_H

#include <atomic>
#include <cstring>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/dim.h"
#include "core/storage.h"
#include "core/shape_utils.h"
#include "util/types.h"
using feather::BufferPool;
using feather::DataType;

#define BYTE_LEN sizeof(float)

namespace feather {

class Tensor {
   public:
    Tensor() {}
    explicit Tensor(std::shared_ptr<Buffer> buffer) : m_storage(std::make_shared<Storage>(std::move(buffer))) {}

    explicit Tensor(const size_t data_size) {
        m_memory_size = data_size;
        m_storage = DeferredAllocationScope::Enabled() ? std::make_shared<Storage>(data_size)
                      : std::make_shared<Storage>(BufferPool::getInstance()->allocate(data_size));
    }

    explicit Tensor(const std::vector<int64_t> &shape) {
        m_dims.ConstructFrom(shape);
        size_t data_size = 0;
        CHECK(CheckedByteCount(shape, BYTE_LEN, &data_size));
        m_memory_size = data_size;
        m_storage = DeferredAllocationScope::Enabled() ? std::make_shared<Storage>(data_size)
                      : std::make_shared<Storage>(BufferPool::getInstance()->allocate(data_size));
    }

    template <typename T>
    void Assign(const std::vector<T> &data, const std::vector<int64_t> &shape) {
        m_dims.ConstructFrom(shape);
        size_t data_size = data.size() * sizeof(T);
        size_t count = 0;
        CHECK(CheckedElementCount(shape, &count));
        CHECK(data.size() == count);
        if (!HasCapacity(data_size)) {
            ResetBuffer(std::make_shared<Buffer>(data_size), data_size);
        }
        auto *dst = mutable_data<T>();
        if (data_size != 0) memcpy(dst, data.data(), data_size);
    }

    template <typename T, typename DDim>
    void Assign(const T *data, const DDim &dim) {
        Resize(dim);
        size_t bytes = 0;
        CHECK(CheckedByteCount(dim.data(), sizeof(T), &bytes));
        if (!HasCapacity(bytes)) {
            ResetBuffer(std::make_shared<Buffer>(bytes), bytes);
        }
        auto *dst = mutable_data<T>();
        if (bytes != 0) memcpy(dst, data, bytes);
    }

    template <typename T>
    const T *data() const {
        size_t bytes = 0;
        CHECK(CheckedByteCount(m_dims.data(), sizeof(T), &bytes));
        CHECK(HasCapacity(bytes));
        return reinterpret_cast<const T *>(static_cast<const char *>(m_storage->data()) + m_offset);
    }

    void Resize(const DDim &dim) { m_dims = dim; }
    void Resize(const std::vector<int64_t> &x) { m_dims.ConstructFrom(x); }

    const DDim &dims() const { return m_dims; }
    int64_t numel() const {
        size_t count = 0;
        CHECK(CheckedElementCount(m_dims.data(), &count));
        return static_cast<int64_t>(count);
    }

    DataType data_type() const { return m_data_type; }
    void set_data_type(const DataType &data_type) { m_data_type = data_type; }
    const QuantizationParams& quantization() const { return m_quantization; }
    void set_quantization(const QuantizationParams& quantization);
    float quantization_scale() const { return m_quantization.enabled ? m_quantization.scale : 1.0f; }
    DataLayout layout() const { return m_layout; }
    void set_layout(DataLayout layout) { m_layout = layout; }
    bool is_immutable() const { return m_immutable; }
    void set_immutable(bool immutable) { m_immutable = immutable; }
    uint64_t mutation_version() const { return m_storage ? m_storage->version() : 0; }

    template <typename T>
    T *mutable_data() {
        m_data_type = DataTypeTrait<T>::type();
        size_t bytes = 0;
        CHECK(CheckedByteCount(m_dims.data(), sizeof(T), &bytes));
        CHECK(HasCapacity(bytes));
        EnsureWritable();
        m_memory_size = bytes;
        MarkStorageMutated();
        return reinterpret_cast<T *>(static_cast<char *>(m_storage->MutableData()) + m_offset);
    }

    template <typename T>
    T *mutable_data(size_t memory_size) {
        m_data_type = DataTypeTrait<T>::type();
        CHECK(HasCapacity(memory_size));
        EnsureWritable();
        m_memory_size = memory_size;
        MarkStorageMutated();
        return reinterpret_cast<T *>(static_cast<char *>(m_storage->MutableData()) + m_offset);
    }

    void *mutable_data(size_t memory_size);

    // Reads never detach shared weights or mark storage as mutated. Writers
    // must use mutable_data(), which enforces copy-on-write and versioning.
    const void *raw_data() const { return static_cast<const char *>(m_storage->data()) + m_offset; }

    void clear() {
        m_storage.reset();
        m_offset = 0;
        m_memory_size = 0;
    }

    size_t data_size() const { return this->dims().production(); }

    size_t memory_size() const { return m_memory_size; }

    size_t offset() const { return m_offset; }

    bool IsInitialized() const { return m_storage != nullptr && m_storage->available(); }

    // Other share data to this.
    void ShareDataWith(const Tensor &other);

    // Exchanges the backing storage while preserving each tensor's shape and
    // semantic metadata. Callers must ensure the tensors are shape-compatible.
    void SwapStorage(Tensor& other);

    std::shared_ptr<const Storage> storage() const { return m_storage; }
    void ResetStorage(std::shared_ptr<const Storage> storage, size_t memory_size, size_t offset = 0);

    void ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size);
    void ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size, size_t offset);

    template <typename T>
    Tensor Slice(int64_t begin, int64_t end) const;

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
        std::vector<int> split_idxs;
        int base = 1;
        for (int i = tensor.dims().size() - 1; i >= 0; i--) {
            base *= tensor.dims()[i];
            split_idxs.push_back(base);
        }

        os << "Tensor:" << '\n';
        os << "dim: " << tensor.dims() << '\n';
        for (int i = 0; i < tensor.dims().size(); i++) {
            os << "[";
        }
        for (int i = 0; i < tensor.dims().production(); i++) {
            os << tensor.template data<float>()[i] << ", ";
            std::string s = ", ";
            for (int j = 0; j < split_idxs.size(); j++) {
                if ((i + 1) % split_idxs[j] == 0) {
                    s = "]" + s;
                    if (tensor.dims().production() - i > 1) {
                        s = s + "[";
                    }
                }
            }
            os << "\b\b" + s;
        }
        os << "\b\b  \n";
        return os;
    }

   private:
    bool HasCapacity(size_t bytes) const {
        return m_storage != nullptr && m_offset <= m_storage->size() && bytes <= m_storage->size() - m_offset;
    }
    void EnsureWritable();
    void MarkStorageMutated();

    DDim m_dims;
    DataType m_data_type{DataType::UNKNOWN};
    DataLayout m_layout{DataLayout::ND};
    std::shared_ptr<const Storage> m_storage;
    size_t m_memory_size{};
    size_t m_offset{};
    bool m_immutable{false};
    QuantizationParams m_quantization{};
};

}  // namespace feather

#endif  // FEATHER_INFER_DATA_TENSOR_H
