// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/operator/non_max_suppression_op.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

#include "core/shape_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {
struct Box {
    double y1, x1, y2, x2;
};

Box ReadBox(const float* value, int64_t center) {
    if (center == 0) {
        return {std::min(value[0], value[2]), std::min(value[1], value[3]),
                std::max(value[0], value[2]), std::max(value[1], value[3])};
    }
    const double x = value[0], y = value[1], w = value[2], h = value[3];
    return {y - h / 2, x - w / 2, y + h / 2, x + w / 2};
}

double IoU(const Box& a, const Box& b) {
    const double intersection = std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1)) *
                                std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const double area_a = std::max(0.0, a.y2 - a.y1) * std::max(0.0, a.x2 - a.x1);
    const double area_b = std::max(0.0, b.y2 - b.y1) * std::max(0.0, b.x2 - b.x1);
    const double total = area_a + area_b - intersection;
    return total > 0.0 ? intersection / total : 0.0;
}

class CommonNonMaxSuppressionKernel final : public KernelBase {
 public:
    int32_t compute() override {
        AutoTimer timer("Common::NonMaxSuppression::FP32");
        const auto* p = ParamAs<operators::NmsParam>();
        int64_t maximum = 0;
        float iou = 0, threshold = 0;
        if (!p || !operators::ReadNmsControls(*p, &maximum, &iou, &threshold) ||
            !operators::PrepareNmsOutput(*p)) return -1;
        const int64_t batches = p->boxes->dims()[0];
        const int64_t count = p->boxes->dims()[1];
        const int64_t classes = p->scores->dims()[1];
        std::vector<int64_t> result;
        if (maximum == 0 || batches == 0 || count == 0 || classes == 0)
            return operators::FinishNmsOutput(*p, result) ? 0 : -1;
        size_t box_bytes = 0, score_bytes = 0;
        if (!CheckedByteCount(p->boxes->dims().data(), sizeof(float), &box_bytes) ||
            !CheckedByteCount(p->scores->dims().data(), sizeof(float), &score_bytes)) return -1;
        auto ready = [](const std::shared_ptr<Tensor>& t, size_t bytes) {
            return t->IsInitialized() && t->storage() && t->offset() <= t->storage()->size() &&
                   bytes <= t->storage()->size() - t->offset();
        };
        if (!ready(p->boxes, box_bytes) || !ready(p->scores, score_bytes)) return -1;
        const auto* boxes = p->boxes->data<float>();
        const auto* scores = p->scores->data<float>();
        maximum = std::min(maximum, count);
        for (int64_t batch = 0; batch < batches; ++batch) {
            std::vector<Box> decoded(static_cast<size_t>(count));
            for (int64_t index = 0; index < count; ++index)
                decoded[static_cast<size_t>(index)] = ReadBox(boxes + (batch * count + index) * 4, p->center_point_box);
            for (int64_t category = 0; category < classes; ++category) {
                const float* class_scores = scores + (batch * classes + category) * count;
                std::vector<int64_t> candidates;
                for (int64_t index = 0; index < count; ++index) {
                    bool finite = !std::isnan(class_scores[index]);
                    for (int coordinate = 0; coordinate < 4; ++coordinate)
                        finite = finite && std::isfinite(boxes[(batch * count + index) * 4 + coordinate]);
                    if (finite && class_scores[index] > threshold) candidates.push_back(index);
                }
                std::stable_sort(candidates.begin(), candidates.end(), [&](int64_t a, int64_t b) {
                    return class_scores[a] > class_scores[b];
                });
                std::vector<int64_t> selected;
                for (int64_t index : candidates) {
                    bool suppressed = false;
                    for (int64_t previous : selected) {
                        if (IoU(decoded[static_cast<size_t>(index)], decoded[static_cast<size_t>(previous)]) > iou) {
                            suppressed = true;
                            break;
                        }
                    }
                    if (suppressed) continue;
                    selected.push_back(index);
                    result.insert(result.end(), {batch, category, index});
                    if (static_cast<int64_t>(selected.size()) >= maximum) break;
                }
            }
        }
        return operators::FinishNmsOutput(*p, result) ? 0 : -1;
    }
};

bool g_registered = [] {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "NonMaxSuppression", [] {
        return std::make_unique<CommonNonMaxSuppressionKernel>();
    });
    return true;
}();
}  // namespace

void EnsureCommonNonMaxSuppressionKernelsRegistered() { (void)g_registered; }
}  // namespace kernel
}  // namespace feather
