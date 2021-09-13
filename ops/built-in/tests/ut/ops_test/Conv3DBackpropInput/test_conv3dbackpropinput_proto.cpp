#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"


class Conv3DBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropInput Proto Test TearDown" << std::endl;
  }
};

// Base_Pass_Case
TEST_F(Conv3DBackpropInputProtoTest, Base_Pass_Case){
    ge::op::Conv3DBackpropInput op;

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{2 ,2 ,16 ,16 ,16};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCDHW, ge::DT_INT32);
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

    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 2, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 2, 16, 16, 16}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Same_Padding_Pass_Case
TEST_F(Conv3DBackpropInputProtoTest, Same_Padding_Pass_Case){
    ge::op::Conv3DBackpropInput op;

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{2 ,2 ,16 ,16 ,16};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCDHW, ge::DT_INT32);
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

    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 2, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 2, 16, 16, 16}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// No_Input_Size_Failed
TEST_F(Conv3DBackpropInputProtoTest, No_Input_Size_Failed){
    ge::op::Conv3DBackpropInput op;

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{2 ,2 ,16 ,16 ,16};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCDHW, ge::DT_INT32);
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

    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 2, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 2, 16, 16, 16}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "SAME");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// No_Input_Size
TEST_F(Conv3DBackpropInputProtoTest, No_Input_Size){
    ge::op::Conv3DBackpropInput op;

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));

    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 2, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 2, 16, 16, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}

// check filter size
TEST_F(Conv3DBackpropInputProtoTest, conv3dbackpropinputVerifyFilterSize) {
    ge::op::Conv3DBackpropInput op;
    op.UpdateInputDesc("filter",
                       create_desc_with_ori(
                           {1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {1, 1, 16, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range(
                           {-1, 8, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {-1, 8, 8, 8, 16}, ge::FORMAT_NDHWC,
                           {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    op.UpdateOutputDesc("y",
                        create_desc_shape_range(
                            {-1, 8, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                            {-1, 8, 8, 8, 16}, ge::FORMAT_NDHWC,
                            {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    auto input_size_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NDHWC, ge::DT_INT32);
    input_size_data.update_input_desc_x(ori_tensorDesc);
    input_size_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(input_size_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// get stride length fail
TEST_F(Conv3DBackpropInputProtoTest, conv3dbackpropinputInferStrideLenFail) {
    ge::op::Conv3DBackpropInput op;
    op.UpdateInputDesc("filter",
                       create_desc_with_ori(
                           {1, 1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {1, 1, 1, 16, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range(
                           {-1, -1, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {-1, -1, 8, 8, 16}, ge::FORMAT_NDHWC,
                           {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    op.UpdateOutputDesc("y",
                        create_desc_shape_range(
                            {-1, -1, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                            {-1, -1, 8, 8, 16}, ge::FORMAT_NDHWC,
                            {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    auto input_size_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{-2};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NDHWC, ge::DT_INT32);
    input_size_data.update_input_desc_x(ori_tensorDesc);
    input_size_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(input_size_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    op.SetAttr("strides", {1, 1, 1, 1});
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// get Pad Same Dynamic
TEST_F(Conv3DBackpropInputProtoTest, conv3dbackpropinputInferPadSame) {
    ge::op::Conv3DBackpropInput op;
    op.UpdateInputDesc("filter",
                       create_desc_with_ori(
                           {1, 1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {1, 1, 1, 16, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range(
                           {-1, -1, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                           {-1, -1, 8, 8, 16}, ge::FORMAT_NDHWC,
                           {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    op.UpdateOutputDesc("y",
                        create_desc_shape_range(
                            {-1, -1, 8, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                            {-1, -1, 8, 8, 16}, ge::FORMAT_NDHWC,
                            {{1, 5}, {8, 8}, {8, 8}, {8, 8}, {16, 16}}));
    auto input_size_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{-2};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NDHWC, ge::DT_INT32);
    input_size_data.update_input_desc_x(ori_tensorDesc);
    input_size_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(input_size_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    op.SetAttr("padding", "SAME");
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}