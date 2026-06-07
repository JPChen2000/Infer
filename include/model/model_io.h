#ifndef FEATHER_MODEL_MODEL_IO_H
#define FEATHER_MODEL_MODEL_IO_H

#include <memory>
#include <string>
#include <unordered_map>

#include "core/tensor.h"
#include "model/model_format.h"
#include "model/weight_store.h"

namespace feather {
namespace model {

class ModelWriter {
   public:
    bool Save(const std::string& path, const ModelDesc& model,
              const std::unordered_map<std::string, std::shared_ptr<Tensor>>& weights);
};

class ModelLoader {
   public:
    bool Load(const std::string& path);

    const ModelDesc& model() const { return model_; }
    std::shared_ptr<Tensor> CreateWeightTensor(const std::string& tensor_name);

   private:
    const ValueDesc* FindValue(const std::string& tensor_name) const;

    std::string path_;
    ModelDesc model_;
    WeightStore weight_store_;
};

}  // namespace model
}  // namespace feather

#endif  // FEATHER_MODEL_MODEL_IO_H
