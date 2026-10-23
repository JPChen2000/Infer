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

#include <iostream>
#include <atomic>
#include <cmath>
#include "core/tensor.h"
#include "util/logger.h"


namespace feather {
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

void Tensor::MarkStorageMutated() { if (m_storage) m_storage->MarkMutated(); }

void Tensor::ShareDataWith(const Tensor &other) {
  m_storage = other.m_storage;
  m_dims = other.m_dims;
  m_data_type = other.m_data_type;
  m_layout = other.m_layout;
  m_quantization = other.m_quantization;
  m_memory_size = other.m_memory_size;
  m_immutable = other.m_immutable;
  m_offset = other.m_offset;

}

void Tensor::SwapStorage(Tensor& other) {
  std::swap(m_storage, other.m_storage);
  std::swap(m_memory_size, other.m_memory_size);
  std::swap(m_offset, other.m_offset);
  MarkStorageMutated();
  other.MarkStorageMutated();
}


void Tensor::EnsureWritable() {
  if (m_storage != nullptr && m_storage->read_only()) {
    CHECK(HasCapacity(m_memory_size));
    // Copy the allocation once for every view in this alias group. Session
    // groups have distinct Storage identities while sharing read-only Buffers.
    m_storage->MakeWritable();
    m_immutable = false;
  }
}

void *Tensor::mutable_data(size_t memory_size) {
  CHECK(HasCapacity(memory_size));
  EnsureWritable();
  m_memory_size = memory_size;
  MarkStorageMutated();
  return static_cast<char *>(m_storage->MutableData()) + m_offset;
}

void Tensor::ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size) {
  ResetBuffer(std::move(buffer), memory_size, 0);
}

void Tensor::ResetBuffer(std::shared_ptr<Buffer> buffer, size_t memory_size, size_t offset) {
  CHECK(buffer != nullptr);
  CHECK(offset <= buffer->size());
  CHECK(memory_size <= buffer->size() - offset);
  ResetStorage(std::make_shared<Storage>(std::move(buffer)), memory_size, offset);
}

void Tensor::ResetStorage(std::shared_ptr<const Storage> storage, size_t memory_size, size_t offset) {
  CHECK(storage != nullptr);
  CHECK(offset <= storage->size() && memory_size <= storage->size() - offset);
  m_storage = std::move(storage);
  m_memory_size = memory_size;
  m_offset = offset;
}

} // namespace feather
