#include <gtest/gtest.h>

#include "model/layout_transform.h"

TEST(model_layout_transform_test, ConvertModelToNhwcRewritesImageShapesAndAxes) {
    feather::model::ModelDesc model;
    model.graph.inputs = {"images"};
    model.graph.outputs = {"out"};

    feather::model::ValueDesc input;
    input.tensor.name = "images";
    input.tensor.dims = {1, 3, 8, 8};

    feather::model::ValueDesc resize_out;
    resize_out.tensor.name = "resize_out";
    resize_out.tensor.dims = {1, 16, 16, 16};

    feather::model::ValueDesc concat_out;
    concat_out.tensor.name = "concat_out";
    concat_out.tensor.dims = {1, 32, 16, 16};

    feather::model::ValueDesc reshape_out;
    reshape_out.tensor.name = "reshape_out";
    reshape_out.tensor.dims = {1, 3, 85, 16, 16};

    feather::model::NodeDesc resize;
    resize.name = "resize";
    resize.op_type = "Resize";
    resize.inputs = {"images"};
    resize.outputs = {"resize_out"};
    resize.attributes["scales"] = std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f};

    feather::model::NodeDesc concat;
    concat.name = "concat";
    concat.op_type = "Concat";
    concat.inputs = {"resize_out", "resize_out"};
    concat.outputs = {"concat_out"};
    concat.attributes["axis"] = int64_t{1};

    feather::model::NodeDesc reshape;
    reshape.name = "reshape";
    reshape.op_type = "Reshape";
    reshape.inputs = {"concat_out"};
    reshape.outputs = {"reshape_out"};
    reshape.attributes["shape"] = std::vector<int64_t>{1, 3, 85, 16, 16};

    feather::model::NodeDesc transpose;
    transpose.name = "transpose";
    transpose.op_type = "Transpose";
    transpose.inputs = {"reshape_out"};
    transpose.outputs = {"transpose_out"};
    transpose.attributes["perm"] = std::vector<int64_t>{0, 1, 3, 4, 2};

    model.graph.values = {input, resize_out, concat_out, reshape_out};
    model.graph.nodes = {resize, concat, reshape, transpose};

    ASSERT_TRUE(feather::model::ConvertModelToNhwcInPlace(&model));

    EXPECT_EQ(model.graph.values[0].tensor.dims, std::vector<int64_t>({1, 8, 8, 3}));
    EXPECT_EQ(model.graph.values[0].tensor.layout, feather::DataLayout::NHWC);
    EXPECT_EQ(model.graph.values[1].tensor.dims, std::vector<int64_t>({1, 16, 16, 16}));
    EXPECT_EQ(model.graph.values[1].tensor.layout, feather::DataLayout::NHWC);
    EXPECT_EQ(std::get<std::vector<float>>(model.graph.nodes[0].attributes["scales"]),
              std::vector<float>({1.0f, 2.0f, 2.0f, 1.0f}));
    EXPECT_EQ(std::get<int64_t>(model.graph.nodes[1].attributes["axis"]), 3);
    EXPECT_EQ(std::get<std::vector<int64_t>>(model.graph.nodes[2].attributes["shape"]),
              std::vector<int64_t>({1, 16, 16, 3, 85}));
    EXPECT_EQ(model.graph.values[3].tensor.dims, std::vector<int64_t>({1, 16, 16, 3, 85}));
    EXPECT_EQ(std::get<std::vector<int64_t>>(model.graph.nodes[3].attributes["perm"]),
              std::vector<int64_t>({0, 3, 1, 2, 4}));
}
