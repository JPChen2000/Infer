#ifndef FEATHER_MODEL_LAYOUT_TRANSFORM_H
#define FEATHER_MODEL_LAYOUT_TRANSFORM_H

#include "model/model_format.h"

namespace feather {
namespace model {

bool ConvertModelToNhwcInPlace(ModelDesc* model);

}  // namespace model
}  // namespace feather

#endif  // FEATHER_MODEL_LAYOUT_TRANSFORM_H
