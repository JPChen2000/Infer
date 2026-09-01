#include <iostream>
#include <atomic>
#include <cmath>
#include "core/tensor.h"
#include "util/logger.h"


namespace feather {
namespace {

uint64_t NextMutationVersion() {
  static std::atomic<uint64_t> next_version{1};
  return next_version.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

void Tensor::set_quantization(const QuantizationParams& quantization) {
  m_quantization = quantization;
  if (!m_quantization.scales.empty()) {
    m_quantization.scale = m_quantization.scales.front();
  }
  if (!(m_quantization.scale > 0.0f) || !std::isfinite(m_quantization.scale)) {
    m_quantization.scale = 1.0f;
  }
  if (m_quantization.scales.empty()) {
    m_quantization.scales.push_back(m_quantization.scale);
  } else {
    m_quantization.scales.front() = m_quantization.scale;
  }
  if (!m_quantization.zero_points.empty()) {
    m_quantization.zero_point = m_quantization.zero_points.front();
  }
  if (m_quantization.zero_points.empty()) {
    m_quantization.zero_points.push_back(m_quantization.zero_point);
  } else {
    m_quantization.zero_points.front() = m_quantization.zero_point;
  }
}

void Tensor::MarkStorageMutated() {
  if (m_mutation_state == nullptr) {
    m_mutation_state = std::make_shared<MutationState>();
  }
  m_mutation_state->version.store(NextMutationVersion(), std::memory_order_relaxed);
}

void Tensor::ResetStorageMutationState() {
  m_mutation_state = std::make_shared<MutationState>();
  MarkStorageMutated();
}

void Tensor::ShareDataWith(const Tensor &other) {
  m_buffer = other.m_buffer;
  m_dims = other.m_dims;
  m_data_type = other.m_data_type;
  m_layout = other.m_layout;
  m_quantization = other.m_quantization;
  m_memory_size = other.m_memory_size;
  m_immutable = other.m_immutable;
  m_offset = other.m_offset;
  m_mutation_state = other.m_mutation_state;
  if (m_mutation_state == nullptr) {
    ResetStorageMutationState();
  }
}

void Tensor::SwapStorage(Tensor& other) {
  std::swap(m_buffer, other.m_buffer);
  std::swap(m_memory_size, other.m_memory_size);
  std::swap(m_offset, other.m_offset);
  std::swap(m_mutation_state, other.m_mutation_state);
  MarkStorageMutated();
  other.MarkStorageMutated();
}


void *Tensor::mutable_data(size_t memory_size) {
  m_memory_size = memory_size;
  CHECK(memory_size <= m_buffer->size());
  MarkStorageMutated();
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
  ResetStorageMutationState();
}

void Tensor::ResetBuffer(std::shared_ptr<Buffer> buffer,
                             size_t memory_size, size_t offset) {
  if (m_buffer != nullptr) {
    m_buffer->deallocate();
  }
  m_buffer = buffer;
  m_memory_size = memory_size;
  m_offset = offset;
  ResetStorageMutationState();
}

} // namespace feather
