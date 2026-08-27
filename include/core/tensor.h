#ifndef FEATHER_INFER_DATA_TENSOR_H
#define FEATHER_INFER_DATA_TENSOR_H

#include <atomic>
#include <cstring>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/dim.h"
#include "core/memory.h"
#include "util/types.h"
using feather::BufferPool;
using feather::DataType;

#define BYTE_LEN sizeof(float)

namespace feather {

class Tensor {
   private:
    struct MutationState {
        std::atomic<uint64_t> version{0};
    };

   public:
    Tensor() {}
    explicit Tensor(std::shared_ptr<Buffer> buffer) : m_buffer(buffer) {}

    explicit Tensor(const size_t data_size) {
        m_memory_size = data_size;
        m_buffer = BufferPool::getInstance()->allocate(data_size);
    }

    explicit Tensor(const std::vector<int64_t> &shape) {
        m_dims.ConstructFrom(shape);
        size_t data_size = m_dims.production() * BYTE_LEN;
        m_memory_size = data_size;
        m_buffer = BufferPool::getInstance()->allocate(data_size);
    }

    template <typename T>
    void Assign(const std::vector<T> &data, const std::vector<int64_t> &shape) {
        m_dims.ConstructFrom(shape);
        size_t data_size = data.size() * sizeof(T);
        CHECK(data.size() == m_dims.production());
        if (m_buffer == nullptr || m_buffer->size() < data_size) {
            m_buffer = std::make_shared<Buffer>(data_size);
            ResetStorageMutationState();
        }
        auto *dst = mutable_data<T>();
        memcpy(dst, data.data(), data_size);
    }

    template <typename T, typename DDim>
    void Assign(const T *data, const DDim &dim) {
        Resize(dim);
        if (m_buffer == nullptr || m_buffer->size() < dim.production() * sizeof(T)) {
            m_buffer = std::make_shared<Buffer>(dim.production() * sizeof(T));
            ResetStorageMutationState();
        }
        auto *dst = mutable_data<T>();
        memcpy(dst, data, dim.production() * sizeof(T));
    }

    template <typename T>
    const T *data() const {
        return reinterpret_cast<const T *>(static_cast<char *>(m_buffer->data()) + m_offset);
    }

    void Resize(const DDim &dim) { m_dims = dim; }
    void Resize(const std::vector<int64_t> &x) { m_dims.ConstructFrom(x); }

    const DDim &dims() const { return m_dims; }
    int64_t numel() const { return m_dims.production(); }

    DataType data_type() const { return m_data_type; }
    void set_data_type(const DataType &data_type) { m_data_type = data_type; }
    const QuantizationParams& quantization() const { return m_quantization; }
    void set_quantization(const QuantizationParams& quantization);
    float quantization_scale() const { return m_quantization.enabled ? m_quantization.scale : 1.0f; }
    DataLayout layout() const { return m_layout; }
    void set_layout(DataLayout layout) { m_layout = layout; }
    bool is_immutable() const { return m_immutable; }
    void set_immutable(bool immutable) { m_immutable = immutable; }
    uint64_t mutation_version() const { return m_mutation_state->version.load(std::memory_order_relaxed); }

    template <typename T>
    T *mutable_data() {
        m_data_type = DataTypeTrait<T>::type();
        m_memory_size = m_dims.production() * sizeof(T);
        CHECK(m_memory_size <= m_buffer->size());
        MarkStorageMutated();
        return reinterpret_cast<T *>(static_cast<char *>(m_buffer->data()) + m_offset);
    }

    template <typename T>
    T *mutable_data(size_t memory_size) {
        m_data_type = DataTypeTrait<T>::type();
        m_memory_size = memory_size;
        CHECK(m_memory_size <= m_buffer->size());
        MarkStorageMutated();
        return reinterpret_cast<T *>(static_cast<char *>(m_buffer->data()) + m_offset);
    }

    void *mutable_data(size_t memory_size);

    const void *raw_data() const { return static_cast<char *>((static_cast<char *>(m_buffer->data()) + m_offset)); }

    void *raw_data() { return static_cast<char *>((static_cast<char *>(m_buffer->data()) + m_offset)); }

    void clear() {
        m_buffer->deallocate();
        m_offset = 0;
        MarkStorageMutated();
    }

    size_t data_size() const { return this->dims().production(); }

    size_t memory_size() const { return m_memory_size; }

    size_t offset() const { return m_offset; }

    bool IsInitialized() const { return m_buffer != nullptr && m_buffer->data() != nullptr; }

    // Other share data to this.
    void ShareDataWith(const Tensor &other);

    // Exchanges the backing storage while preserving each tensor's shape and
    // semantic metadata. Callers must ensure the tensors are shape-compatible.
    void SwapStorage(Tensor& other);

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
    void MarkStorageMutated();
    void ResetStorageMutationState();

    DDim m_dims;
    DataType m_data_type{DataType::UNKNOWN};
    DataLayout m_layout{DataLayout::ND};
    std::shared_ptr<Buffer> m_buffer;
    size_t m_memory_size{};
    size_t m_offset{};
    bool m_immutable{false};
    QuantizationParams m_quantization{};
    std::shared_ptr<MutationState> m_mutation_state = std::make_shared<MutationState>();
};

}  // namespace feather

#endif  // FEATHER_INFER_DATA_TENSOR_H
