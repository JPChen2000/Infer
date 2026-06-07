
#include <iostream>
#include <sstream>
#include <vector>
#include "core/dim.h"

namespace feather {
using value_type = int64_t;

value_type DDim::production() const {
    value_type res = 1;
    for (size_t i = 0; i < m_data.size(); i++) {
        res *= m_data[i];
    }
    return res;
}

value_type DDim::count(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int>(m_data.size()));
  if (end < start) {
    return 0;
  }
  value_type sum = 1;
  for (auto i = start; i < end; ++i) {
    sum *= m_data[i];
  }
  return sum;
}

DDim DDim::Slice(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int>(m_data.size()));
  std::vector<value_type> new_dim(end - start);
  for (int i = start; i < end; i++) {
    new_dim[i - start] = m_data[i];
  }
  return DDim(new_dim);
}

std::string DDim::repr() const {
  std::stringstream ss;
  if (empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < this->size() - 1; i++) {
    ss << (*this)[i] << ",";
  }
  if (!this->empty()) ss << (*this)[size() - 1];
  ss << "}";
  return ss.str();
}

} // namespace feather
