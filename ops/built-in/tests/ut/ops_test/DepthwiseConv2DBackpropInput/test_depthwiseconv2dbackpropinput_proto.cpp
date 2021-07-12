#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"


class DepthwiseConv2dBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2dBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2dBackpropInput Proto Test TearDown" << std::endl;
  }
};

// fix Const
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2dBackpropInputFix){
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_with_ori({1, 32, 31, 31}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 31, 31}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("input_grad", create_desc_with_ori({1, 32, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 64, 64}, ge::FORMAT_NCHW));
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 32, 64, 64};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_input_size(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "VALID");
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic n Const range -1
TEST_F(DepthwiseConv2dBackpropInputProtoTest, Base_Pass_Case){
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, 64, 64},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, 64, 64},
                                               ge::FORMAT_NCHW,
                                               {{1, -1}, {32, 32}, {63, 65}, {63, 65}}));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("input_grad",
                    create_desc_shape_range({-1, 32, 64, 64},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, 64, 64},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {63, 65}, {63, 65}}));
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 32, 64, 64};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_input_size(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("input_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// dynamic nhw SAME Var
TEST_F(DepthwiseConv2dBackpropInputProtoTest, Input_Size){
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 3}, {32, 32}, {63, 65}, {63, 65}}));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));

    auto depthwise_conv2d_backprop_input_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    depthwise_conv2d_backprop_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    depthwise_conv2d_backprop_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(depthwise_conv2d_backprop_input_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    op.UpdateOutputDesc("input_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {63, 65}, {63, 65}}));
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


// dynamic nwc VALID Var
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2dBackpropInputDynamicNWC) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("input_size");
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 10}, {16, 32}, {24, 24}, {6, -1}};
    input_sizes_desc->SetValueRange(value_range);
    
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut outbackprop shape [-2]
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2dBackpropInputDynamicRank) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-2},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-2},
                                               ge::FORMAT_NCHW,
                                               {{}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {1, -1}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic general ut with dilations<0
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2dbackpropinputGeneWithDilation) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", 
                       create_desc_with_ori({1, 32, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("input_grad", 
                       create_desc_with_ori({1, 32, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 64, 64}, ge::FORMAT_NCHW));
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 32, 64, 64};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_input_size(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// dtype
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyDtypeTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_INT8, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// strides
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyStridesDimTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyNoStridesTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyDilationsDimTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// data_format
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyDataFormatTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "HWCN");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// padding
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyPaddingTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "LIST");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pads
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyPadsDimTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pads
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2DBackpropInputVerifyPadsTest) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("input_grad", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// fuzzy compile
TEST_F(DepthwiseConv2dBackpropInputProtoTest, DepthwiseConv2dBackpropInputFuzzyCompile) {
    ge::op::DepthwiseConv2DBackpropInput op;
    op.SetAttr("_fuzz_build", true);
    op.UpdateInputDesc("out_backprop",
                       create_desc_with_ori({1, 32, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 24, 24}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({2, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {2, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("input_grad", create_desc_with_ori({1, 16, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                  {1, 16, 24, 24}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 16, 24, 24};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_input_size(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    auto ret = op.InferShapeAndType();
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tesor_desc_x = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::pair<int64_t, int64_t>> input_range;
    tesor_desc_x->GetShapeRange(input_range);
    std::vector<std::pair<int64_t, int64_t>> expect_input_range = {{1, 1}, {32, 32}, {16, 31}, {16, 31}};
    EXPECT_EQ((input_range == expect_input_range), true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
