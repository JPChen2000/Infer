#include "core/memory.h"
#include "util/logger.h"
#include "util/types.h"

namespace feather {

std::shared_ptr<BufferPool> BufferPool::m_buffer_pool = std::make_shared<BufferPool>();
std::mutex BufferPool::m_mtx;

constexpr size_t MAX_CHUNK_SIZE = 8 * 1024;
constexpr size_t MIN_CLASS_SIZE = 64;
constexpr size_t LARGE_ALIGN_SIZE = 4 * 1024;

template <size_t ALIGN_STEP = sizeof(int)>
inline size_t ALIGN_UP(size_t size) {
    return size % ALIGN_STEP == 0 ? size : size - size % ALIGN_STEP + ALIGN_STEP;
}

size_t BufferPool::NormalizeSize(size_t size) {
    if (size == 0) {
        return MIN_CLASS_SIZE;
    }
    if (size <= MAX_CHUNK_SIZE) {
        size_t klass = MIN_CLASS_SIZE;
        while (klass < size) {
            klass <<= 1;
        }
        return klass;
    }
    return ALIGN_UP<LARGE_ALIGN_SIZE>(size);
}

std::shared_ptr<Buffer> BufferPool::allocate(size_t size) {
    std::lock_guard<std::mutex> guard(m_mtx);
    const auto class_size = NormalizeSize(size);

    MemBlock* block = nullptr;
    auto free_it = m_free_blocks.find(class_size);
    if (free_it != m_free_blocks.end() && free_it->second != nullptr) {
        block = free_it->second;
        free_it->second = block->free_next;
        block->free_next = nullptr;
    } else {
        block = allocate_mem_block(class_size);
        block->next = m_mem_cache[class_size];
        m_mem_cache[class_size] = block;
    }

    block->occupied = true;
    std::weak_ptr<BufferPool> weak_pool = m_buffer_pool;
    return std::make_shared<Buffer>(block, [weak_pool](MemBlock* released_block) {
        if (auto pool = weak_pool.lock()) {
            pool->release_block(released_block);
        }
    });
}

MemChunk *BufferPool::allocate_mem_chunk(size_t size) {
    MemChunk *chunk = new MemChunk{};
    chunk->data = new char[size];
    chunk->offset = 0;
    chunk->size = size;
    chunk->next = nullptr;
    // LOG_INFO("allocate_mem_chunk %p", chunk->data);
    return chunk;
}

MemBlock *BufferPool::allocate_mem_block(size_t size) {
    if (size < MAX_CHUNK_SIZE) {
        MemChunk *curr_chunk = m_head_chunk;
        MemChunk *last_chunk = m_head_chunk;

        // allocate block from exist chunk
        while (curr_chunk != nullptr) {
            if (size <= curr_chunk->size - curr_chunk->offset) {
                MemBlock *block = new MemBlock{};
                block->size = size;
                block->data = curr_chunk->data + curr_chunk->offset;
                block->loc = curr_chunk;
                block->occupied = false;
                block->next = nullptr;
                block->free_next = nullptr;
                curr_chunk->offset += ALIGN_UP(size);
                // LOG_INFO("allocate block %p chunk %p", block->data, curr_chunk->data);
                return block;
            }
            last_chunk = curr_chunk;
            curr_chunk = curr_chunk->next;
        }

        // create new chunk
        curr_chunk = allocate_mem_chunk(MAX_CHUNK_SIZE);
        curr_chunk->next = nullptr;
        if (m_head_chunk == nullptr) {
            m_head_chunk = curr_chunk;
        }
        // while is not head chunk
        if (last_chunk != nullptr) {
            last_chunk->next = curr_chunk;
        }
        MemBlock *block = new MemBlock{};
        block->size = size;
        block->data = curr_chunk->data + curr_chunk->offset;
        block->loc = curr_chunk;
        block->occupied = false;
        block->next = nullptr;
        block->free_next = nullptr;
        curr_chunk->offset += ALIGN_UP(size);
        // LOG_INFO("allocate block %p chunk %p", block->data, curr_chunk->data);
        return block;
    } else {
        // block point to sys heap for large buffer
        MemBlock *block = new MemBlock{};
        block->size = size;
        block->data = new char[size];
        block->loc = nullptr;
        block->occupied = false;
        block->next = nullptr;
        block->free_next = nullptr;
        // LOG_INFO("allocate block %p from sys heap", block->data);
        return block;
    }
}

void BufferPool::deallocate(std::shared_ptr<Buffer> &buffer) { buffer->deallocate(); }

void BufferPool::release_block(MemBlock* block) {
    if (block == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> guard(m_mtx);
    if (!block->occupied) {
        return;
    }
    block->occupied = false;
    block->free_next = m_free_blocks[block->size];
    m_free_blocks[block->size] = block;
}

void BufferPool::deallocate_all() {
    m_free_blocks.clear();
    for (auto it = m_mem_cache.begin(); it != m_mem_cache.end(); ++it) {
        auto block = it->second;
        while (block != nullptr) {
            auto next = block->next;
            if (block->loc == nullptr && block->size != 0) {
                delete[] block->data;
            }
            delete block;
            block = next;
        }
    }
    m_mem_cache.clear();

    MemChunk *curr = m_head_chunk;
    while (curr != nullptr) {
        auto next = curr->next;
        delete[] curr->data;
        delete curr;
        curr = next;
    }
    m_head_chunk = nullptr;
}

}  // namespace feather
