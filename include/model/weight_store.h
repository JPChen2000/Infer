#ifndef FEATHER_MODEL_WEIGHT_STORE_H
#define FEATHER_MODEL_WEIGHT_STORE_H

#include <memory>
#include <string>
#include <unordered_map>

#include "core/memory.h"
#include "core/tensor.h"
#include "model/model_format.h"

namespace feather {
namespace model {

class MappedFile {
   public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool Open(const std::string& path);
    void Close();

    void* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }

   private:
    int fd_{-1};
    void* data_{};
    size_t size_{};
    std::string path_;
};

class WeightStore {
   public:
    bool AddShard(const std::string& path);
    std::shared_ptr<Tensor> CreateTensorView(const TensorDesc& desc, const WeightLocation& location);

   private:
    std::shared_ptr<MappedFile> GetOrOpenShard(const std::string& path);

    std::unordered_map<std::string, std::shared_ptr<MappedFile>> shards_;
};

}  // namespace model
}  // namespace feather

#endif  // FEATHER_MODEL_WEIGHT_STORE_H
