#ifndef FEATHER_INFER_DATA_DIM_H
#define FEATHER_INFER_DATA_DIM_H
#include <iostream>
#include <string>
#include <vector>

namespace feather {

class DDim {
   public:
    using value_type = int64_t;

    DDim() = default;

    explicit DDim(const std::vector<value_type> &x) { m_data = x; }
    void ConstructFrom(const std::vector<value_type> &x) { m_data = x; }

    value_type operator[](int offset) const { return m_data[offset]; }
    value_type &operator[](int offset) { return m_data[offset]; }

    size_t size() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }

    value_type production() const;

    const std::vector<value_type> &data() const { return m_data; }
    value_type count(int start, int end) const;

    DDim Slice(int start, int end) const;

    DDim Flatten2D(int col) const {
        return DDim(std::vector<value_type>({Slice(0, col).production(), Slice(col, size()).production()}));
    }
    std::string repr() const;

    friend std::ostream &operator<<(std::ostream &os, const DDim &dims) {
        os << dims.repr();
        return os;
    }

    friend bool operator==(const DDim &a, const DDim &b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    friend bool operator!=(const DDim &a, const DDim &b) {
        if (a.size() != b.size()) return true;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return true;
        }
        return false;
    }

   private:
    std::vector<value_type> m_data;
};
}  // namespace feather

#endif
