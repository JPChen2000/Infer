// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_KERNEL_CUDA_NMS_H
#define FEATHER_KERNEL_CUDA_NMS_H
#include <cstdint>
#include <vector>
namespace feather {
namespace kernel {
namespace cuda_detail {
struct CudaNmsOptions {
    int64_t batches{1}, classes{1}, boxes{0}, max_per_class{0};
    float iou_threshold{0}, score_threshold{0};
    bool center_point_box{false};
    bool score_inclusive{false};
    bool double_iou{false};
    bool preserve_corner_order{false};
};
// Inputs reside on the active inference stream. Boxes [B,N,4], scores
// [B,C,N]; optional labels [B,N] prevent suppression between classes when
// selecting a single globally sorted set (YOLO best-class postprocessing).
// Returns flattened [batch,class,box] triples in batch/class/score order.
int SelectNmsOnCuda(const float* boxes, const float* scores, const int64_t* labels,
                    const CudaNmsOptions& options, std::vector<int64_t>* selected);
}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather
#endif
