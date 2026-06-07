#ifndef FEATHER_CORE_MEMORY_H
#define FEATHER_CORE_MEMORY_H

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util/logger.h"
#include "util/types.h"

namespace feather {

struct MemChunk {
    char* data;
    size_t offset;
    size_t size;
    MemChunk* next;
};

struct MemBlock {
    char* data;
    size_t size;
    MemChunk* loc;

    bool occupied;
    MemBlock* next;
    MemBlock* free_next;
};

class Buffer {
   public:
    Buffer() = default;

    explicit Buffer(size_t size) : m_size(size) {
        m_data = new char[size];
        m_malloc = true;
        // LOG_DEBUG("[Buffer] create buffer from heap");
    }

    explicit Buffer(void* data, size_t size) {
        m_data = data;
        m_size = size;
        m_malloc = false;
    }

    explicit Buffer(void* data, size_t size, std::shared_ptr<void> owner) {
        m_data = data;
        m_size = size;
        m_owner = std::move(owner);
        m_malloc = false;
    }

    explicit Buffer(MemBlock* block) {
        m_block = block;
        m_data = m_block->data;
        m_size = m_block->size;
        m_malloc = false;
    }

    explicit Buffer(MemBlock* block, std::function<void(MemBlock*)> releaser) {
        m_block = block;
        m_data = m_block->data;
        m_size = m_block->size;
        m_releaser = std::move(releaser);
        m_malloc = false;
    }

    ~Buffer() {
        deallocate();
        if (m_malloc) {
            delete[] static_cast<char*>(m_data);
            // LOG_DEBUG("[Buffer] delete buffer from heap");
        }
    }

    void allocate(size_t size) {
        if (m_data == nullptr) {
            m_data = new char[size];
            m_size = size;
            m_malloc = true;
        }
    }

    void deallocate() {
        if (m_malloc == false && m_block != nullptr && !m_released) {
            if (m_releaser) {
                m_releaser(m_block);
            } else {
                m_block->occupied = false;
            }
            m_released = true;
            // LOG_DEBUG("[Buffer] delete buffer from bufferpool");
        }
    }

    void* data() { return m_data; }

    void set_data(void* data, size_t size) {
        m_data = data;
        m_size = size;
        m_malloc = false;
        m_block = nullptr;
        m_releaser = nullptr;
        m_released = false;
    };

    size_t size() { return m_size; }

   private:
    bool m_malloc{};
    bool m_released{};
    void* m_data{};
    size_t m_size{};
    MemBlock* m_block{};
    std::shared_ptr<void> m_owner;
    std::function<void(MemBlock*)> m_releaser;
};

class BufferPool : public std::enable_shared_from_this<BufferPool> {
   public:
    BufferPool() {}
    ~BufferPool() { deallocate_all(); }

    static std::shared_ptr<BufferPool>& getInstance() { return m_buffer_pool; }

    std::shared_ptr<Buffer> allocate(size_t size);
    void deallocate(std::shared_ptr<Buffer>& buffer);
    void deallocate_all();

   private:
    static size_t NormalizeSize(size_t size);
    void release_block(MemBlock* block);
    MemBlock* allocate_mem_block(size_t size);
    MemChunk* allocate_mem_chunk(size_t size);

   private:
    static std::mutex m_mtx;
    static std::shared_ptr<BufferPool> m_buffer_pool;
    std::unordered_map<size_t, MemBlock*> m_mem_cache;
    std::unordered_map<size_t, MemBlock*> m_free_blocks;
    MemChunk* m_head_chunk{};
};

}  // namespace feather

#endif  // FEATHER_CORE_MEMORY_H
