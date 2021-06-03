#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"


class AvgPool3DGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DGradProtoTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DGradProtoTest Proto Test TearDown" << std::endl;
  }
};

// Base_Pass_Case
TEST_F(AvgPool3DGradProtoTest, BaseCase){
    ge::op::AvgPool3DGrad op;
    op.UpdateInputDesc("grads", create_desc_with_ori(
      {16, 2, 3, 6, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, 6, 16}, ge::FORMAT_NDHWC));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_orig_input_size{16 ,12 ,12 ,12 ,1};
    ge::TensorDesc tensor_desc_input_size(ge::Shape({5}),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = dims_orig_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_orig_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("orig_input_shape").set_attr_value(constTensor);
    op.set_input_orig_input_shape(const0);

    delete[] conv_input_size_tensor_value;

    op.UpdateInputDesc("orig_input_shape", tensor_desc_input_size);

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// dynamic base case1---padding SAME
TEST_F(AvgPool3DGradProtoTest, DynamicBaseCase1){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, -1, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,40},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic base case2--padding VALID
TEST_F(AvgPool3DGradProtoTest, DynamicBaseCase2){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {-1, 48, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {-1, 48, 1, -1, -1, 16}, ge::FORMAT_NDHWC,
      {{1,2},{48,48},{1,231},{1,227},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 4, 4, 1});
    op.SetAttr("padding", "VALID");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic base case3---dedy shape is const
TEST_F(AvgPool3DGradProtoTest, DynamicBaseCase3){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, 3, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,3},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic base case4---orig_input_shape is const
TEST_F(AvgPool3DGradProtoTest, DynamicBaseCase4){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, -1, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,3},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {16, 12, 12, 12, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 12, 12, 12, 1}, ge::FORMAT_NDHWC,
      {{16, 16}, {12, 12}, {12, 12}, {12, 12}, {1,1}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic ksize size invalid case
TEST_F(AvgPool3DGradProtoTest, DynamicKsizeInvalid){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, 6, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,40},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic strides invalid case
TEST_F(AvgPool3DGradProtoTest, DynamicStridesInvalid){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, 6, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,40},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic dataformat invalid case
TEST_F(AvgPool3DGradProtoTest, DynamicDataFormatInvalid){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {16, 2, 3, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 1, 3, 6, 16}, ge::FORMAT_NDHWC,
      {{16,16},{2,2},{3,3},{3,40},{1,1}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("data_format", "NNNNN");
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic dedy=-2 case
TEST_F(AvgPool3DGradProtoTest, DynamicDedyInvalid){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {-2}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {-2}, ge::FORMAT_NDHWC,
      {{}, {}, {}, {}, {}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic invalid padding case
TEST_F(AvgPool3DGradProtoTest, DynamicPaddingInvalid){
    ge::op::AvgPool3DGrad op;

    op.UpdateInputDesc("grads", create_desc_shape_range(
      {-2}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {-2}, ge::FORMAT_NDHWC,
      {{}, {}, {}, {}, {}}));
    op.UpdateInputDesc("orig_input_shape", create_desc_shape_range(
      {5}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {5}, ge::FORMAT_NDHWC, {{5, 5}}));

    op.SetAttr("ksize", {1, 1, 5, 1, 1});
    op.SetAttr("strides", {1, 10 ,4, 2, 1});
    op.SetAttr("padding", "NONE");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}