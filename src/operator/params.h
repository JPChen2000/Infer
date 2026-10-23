// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(w);
        visit(bias);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(a);
        visit(b);
        visit(bias);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(w);
        visit(bias);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(state);
        visit(mixed);
        visit(weight);
        visit(conv_out);
        visit(discarded_prefix);
        visit(next_state);
    }
};

struct BatchNormParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> scale;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> mean;
    std::shared_ptr<Tensor> var;
    std::shared_ptr<Tensor> out;
    float epsilon{1e-5f};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(scale);
        visit(bias);
        visit(mean);
        visit(var);
        visit(out);
    }
};

struct UnaryParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct BinaryParam : ParamBase {
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(lhs);
        visit(rhs);
        visit(out);
    }
};

struct MatMulParam : ParamBase {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(a);
        visit(b);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(a);
        visit(b);
        visit(out);
    }
};

struct QwenGatedDeltaStateParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> decay;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(state);
        visit(k);
        visit(v);
        visit(beta);
        visit(decay);
        visit(out);
    }
};

struct QwenGatedDeltaOutputParam : ParamBase {
    std::shared_ptr<Tensor> state;
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(state);
        visit(q);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(state);
        visit(k);
        visit(v);
        visit(beta);
        visit(decay);
        visit(q);
        visit(next_state);
        visit(out);
    }
};

struct QwenRmsNormParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> epsilon;
    std::shared_ptr<Tensor> out;
    float weight_offset{0.0f};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(weight);
        visit(epsilon);
        visit(out);
    }
};

struct ReshapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> target_shape;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(shape);
        visit(out);
    }
};

struct AxesParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> axes_tensor;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(axes_tensor);
        visit(out);
    }
};

struct CastParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    DataType to{DataType::FP32};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct ReduceMeanParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
    bool keepdims{true};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct ReduceSumParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> axes;
    bool keepdims{true};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct GatherParam : ParamBase {
    std::shared_ptr<Tensor> data;
    std::shared_ptr<Tensor> indices;
    std::shared_ptr<Tensor> out;
    int32_t axis{0};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(data);
        visit(indices);
        visit(out);
    }
};

struct EqualParam : ParamBase {
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(lhs);
        visit(rhs);
        visit(out);
    }
};

struct ShapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int64_t start{0};
    int64_t end{std::numeric_limits<int64_t>::max()};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct ConstantOfShapeParam : ParamBase {
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
    DataType output_type{DataType::UNKNOWN};
    int64_t int_value{0};
    float float_value{0.0f};
    bool use_float_value{false};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(shape);
        visit(out);
    }
};

struct ExpandParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> shape;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(shape);
        visit(out);
    }
};

struct WhereParam : ParamBase {
    std::shared_ptr<Tensor> condition;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(condition);
        visit(x);
        visit(y);
        visit(out);
    }
};

struct FlattenParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t axis;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct GlobalAveragePoolParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct ConcatParam : ParamBase {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> out;
    int32_t axis;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(out);
        for (auto& tensor : inputs) visit(tensor);
    }
};

struct SplitParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> split;
    std::vector<std::shared_ptr<Tensor>> outputs;
    int32_t axis;
    std::vector<int64_t> split_sizes;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(split);
        for (auto& tensor : outputs) visit(tensor);
    }
};

struct SoftmaxParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t axis;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
};

struct TransposeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> perm;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(out);
    }
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
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(starts);
        visit(ends);
        visit(axes);
        visit(steps);
        visit(out);
    }
};

struct ResizeParam : ParamBase {
    enum class Mode { kNearest, kLinear };
    enum class CoordinateMode { kAsymmetric, kHalfPixel, kPytorchHalfPixel, kAlignCorners };
    enum class NearestMode { kFloor, kCeil, kRoundPreferFloor, kRoundPreferCeil };
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> roi;
    std::shared_ptr<Tensor> scales_tensor;
    std::shared_ptr<Tensor> sizes;
    std::shared_ptr<Tensor> out;
    std::vector<float> scales;
    // Direct Feather graphs keep the historical YOLO defaults. ONNX import
    // writes its own defaults explicitly.
    Mode mode{Mode::kNearest};
    CoordinateMode coordinate_mode{CoordinateMode::kAsymmetric};
    NearestMode nearest_mode{NearestMode::kFloor};
    std::vector<double> effective_scales;
    bool onnx_nchw_controls{false};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(roi);
        visit(scales_tensor);
        visit(sizes);
        visit(out);
    }
};

struct ResizeConcatParam : ParamBase {
    std::shared_ptr<Tensor> resize_input;
    std::shared_ptr<Tensor> concat_input;
    std::shared_ptr<Tensor> out;
    std::vector<float> scales;
    int32_t axis{1};
    int32_t resize_input_index{0};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(resize_input);
        visit(concat_input);
        visit(out);
    }
};

struct PowParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> exponent_tensor;
    std::shared_ptr<Tensor> out;
    float exponent{1.0f};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(exponent_tensor);
        visit(out);
    }
};

struct YoloDecodeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> xy_scale;
    std::shared_ptr<Tensor> grid;
    std::shared_ptr<Tensor> stride;
    std::shared_ptr<Tensor> wh_scale;
    std::shared_ptr<Tensor> anchor_grid;
    std::shared_ptr<Tensor> out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(xy_scale);
        visit(grid);
        visit(stride);
        visit(wh_scale);
        visit(anchor_grid);
        visit(out);
    }
};

struct QuantizeLinearParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> scale;
    std::shared_ptr<Tensor> zero_point;
    std::shared_ptr<Tensor> out;
    int64_t axis{-1};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(scale);
        visit(zero_point);
        visit(out);
    }
};

struct DequantizeLinearParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> scale;
    std::shared_ptr<Tensor> zero_point;
    std::shared_ptr<Tensor> out;
    DataType to{DataType::FP32};
    int64_t axis{-1};
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(input);
        visit(scale);
        visit(zero_point);
        visit(out);
    }
};

}  // namespace operators
}  // namespace feather
#endif  // FtEATHER_OPERATOR_OP_PARAMS_H
