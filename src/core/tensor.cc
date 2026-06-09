#include <iostream>
#include "core/tensor.h"
#include "util/logger.h"


namespace feather {
void Tensor::ShareDataWith(const Tensor &other) {
  m_buffer = other.m_buffer;
  m_dims = other.m_dims;
  m_data_type = other.m_data_type;
  m_layout = other.m_layout;
  m_memory_size = other.m_memory_size;
  m_offset = other.m_offset;
}


void *Tensor::mutable_data(size_t memory_size) {
  m_memory_size = memory_size;
  CHECK(memory_size <= m_buffer->size());
  return static_cast<char *>(m_buffer->data()) + m_offset;
}

void Tensor::ResetBuffer(std::shared_ptr<Buffer> buffer,
                             size_t memory_size) {
  if (m_buffer != nullptr) {
    m_buffer->deallocate();
  }
  m_buffer = buffer;
  m_memory_size = memory_size;
  m_offset = 0;
}

void Tensor::ResetBuffer(std::shared_ptr<Buffer> buffer,
                             size_t memory_size, size_t offset) {
  if (m_buffer != nullptr) {
    m_buffer->deallocate();
  }
  m_buffer = buffer;
  m_memory_size = memory_size;
  m_offset = offset;
}

} // namespace feather
