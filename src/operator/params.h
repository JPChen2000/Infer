#ifndef FEATHER_OPERATOR_OP_PARAMS_H
#define FEATHER_OPERATOR_OP_PARAMS_H
#include <cstdint>
#include <limits>

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
    float alpha{1.0f};
    float beta{1.0f};
    bool trans_a{false};
    bool trans_b{false};
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

// Qwen's linear-attention convolution consumes a three-token BF16 state and
// one new projected token. It also produces the shifted state for the next
// decode step.
struct QwenDepthwiseConvStateParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> mixed;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> conv_out;
    std::shared_ptr<Tensor> discarded_prefix;
    std::shared_ptr<Tensor> next_state;
    // FP8 state-convolution fusion keeps the BF16 state interface while
    // preserving the quantization points of the original Cast/Conv/Cast
    // sequence.
    DataType fp8_dtype{DataType::UNKNOWN};
    float fp8_input_scale{1.0f};
    float fp8_output_scale{1.0f};
};

struct BatchNormParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> scale;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> mean;
    std::shared_ptr<Tensor> var;
    std::shared_ptr<Tensor> out;
    float epsilon{1e-5f};
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

// Fused Qwen lm-head projection and greedy token selection. The weight is
// stored as [vocabulary, hidden] and the activation is a single [1, hidden]
// row. The output is one INT64 token id instead of a materialized logits row.
struct QwenGemmArgmaxParam : ParamBase {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> out;
    // FP8 lm-head graphs quantize the intermediate Gemm result before the
    // original Cast-to-BF16 node.  Keep that scale explicit so the fused
    // terminal path preserves the exported graph's greedy-selection values.
    float output_scale{1.0f};
};

struct QwenGatedDeltaStateParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> decay;
    std::shared_ptr<Tensor> out;
};

struct QwenGatedDeltaOutputParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> out;
};

struct QwenGatedDeltaParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> decay;
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> next_state;
    std::shared_ptr<Tensor> out;
};

struct QwenRmsNormParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> epsilon;
    std::shared_ptr<Tensor> out;
    float weight_offset{0.0f};
};

struct ReshapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> target_shape;
};

struct AxesParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> axes_tensor;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
};

struct CastParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    DataType to{DataType::FP32};
};

struct ReduceMeanParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
    bool keepdims{true};
};

struct ReduceSumParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
    bool keepdims{true};
};

struct GatherParam : ParamBase {
    std::shared_ptr<Tensor> data;
    std::shared_ptr<Tensor> indices;
    std::shared_ptr<Tensor> out;
    int32_t axis{0};
};

struct EqualParam : ParamBase {
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> out;
};

struct ShapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int64_t start{0};
    int64_t end{std::numeric_limits<int64_t>::max()};
};

struct ConstantOfShapeParam : ParamBase {
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
    DataType output_type{DataType::UNKNOWN};
    int64_t int_value{0};
    float float_value{0.0f};
    bool use_float_value{false};
};

struct ExpandParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
};

struct WhereParam : ParamBase {
    std::shared_ptr<Tensor> condition;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> out;
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

struct GlobalAveragePoolParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
};

struct ConcatParam : ParamBase {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> out;
    int32_t axis;
};

struct SplitParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> split;
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
    std::shared_ptr<Tensor> starts;
    std::shared_ptr<Tensor> ends;
    std::shared_ptr<Tensor> axes;
    std::shared_ptr<Tensor> steps;
    std::shared_ptr<Tensor> out;
    int32_t axis;
    int32_t start;
    int32_t end;
};

struct ResizeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> roi;
    std::shared_ptr<Tensor> scales_tensor;
    std::shared_ptr<Tensor> sizes;
    std::shared_ptr<Tensor> out;
    std::vector<float> scales;
};

struct ResizeConcatParam : ParamBase {
    std::shared_ptr<Tensor> resize_input;
    std::shared_ptr<Tensor> concat_input;
    std::shared_ptr<Tensor> out;
    std::vector<float> scales;
    int32_t axis{1};
    int32_t resize_input_index{0};
};

struct PowParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> exponent_tensor;
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
