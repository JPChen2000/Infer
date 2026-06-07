#ifndef FEATHER_DEMO_DETECTION_H
#define FEATHER_DEMO_DETECTION_H

namespace feather {
namespace demo {

struct Detection {
    int class_id{};
    float score{};
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_DETECTION_H
