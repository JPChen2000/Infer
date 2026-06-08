#ifndef FEATHER_OPERATOR_OP_PARAMS_H
#define FEATHER_OPERATOR_OP_PARAMS_H
#include "core/tensor.h"

namespace feather {
namespace operators {
struct ParamBase {};

struct FcParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> out;
};

struct GemmParam : ParamBase {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> out;
};

struct Conv2dParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> out;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h{1};
    int32_t dilation_w{1};
    int32_t group{1};
};

struct UnaryParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
};

struct BinaryParam : ParamBase {
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> out;
};

struct MatMulParam : ParamBase {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> out;
};

struct ReshapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> target_shape;
};

struct FlattenParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t axis;
};

struct PoolParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
};

struct ConcatParam : ParamBase {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> out;
    int32_t axis;
};

struct SplitParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::vector<std::shared_ptr<Tensor>> outputs;
    int32_t axis;
    std::vector<int64_t> split_sizes;
};

struct SoftmaxParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t axis;
};

struct TransposeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> perm;
};

struct SliceParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t axis;
    int32_t start;
    int32_t end;
};

struct ResizeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<float> scales;
};

struct PowParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    float exponent{1.0f};
};

struct YoloDecodeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> xy_scale;
    std::shared_ptr<Tensor> grid;
    std::shared_ptr<Tensor> stride;
    std::shared_ptr<Tensor> wh_scale;
    std::shared_ptr<Tensor> anchor_grid;
    std::shared_ptr<Tensor> out;
};

}  // namespace operators
}  // namespace feather
#endif  // FtEATHER_OPERATOR_OP_PARAMS_H
