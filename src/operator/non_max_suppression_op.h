// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_OPERATOR_NON_MAX_SUPPRESSION_OP_H
#define FEATHER_OPERATOR_NON_MAX_SUPPRESSION_OP_H
#include "core/operator.h"

namespace feather {
namespace operators {
struct NmsParam {
    std::shared_ptr<Tensor> boxes, scores, max_output, iou_threshold, score_threshold, out;
    int64_t center_point_box{0};
    int64_t capacity_rows{0};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(boxes); visit(scores); visit(max_output); visit(iou_threshold); visit(score_threshold); visit(out);
    }
};
bool ValidateNms(const NmsParam& p);
bool ReadNmsControls(const NmsParam& p, int64_t* maximum, float* iou, float* score);
bool PrepareNmsOutput(const NmsParam& p);
bool FinishNmsOutput(const NmsParam& p, const std::vector<int64_t>& indices);
void EnsureNonMaxSuppressionOperatorRegistered();
}  // namespace operators
namespace kernel {
void EnsureCommonNonMaxSuppressionKernelsRegistered();
void EnsureCudaNonMaxSuppressionKernelsRegistered();
}  // namespace kernel
}  // namespace feather
#endif
