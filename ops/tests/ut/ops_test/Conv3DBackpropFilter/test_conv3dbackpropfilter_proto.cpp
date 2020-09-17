#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "all_ops.h"

class Conv3DBackpropFilterProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropFilter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropFilter Proto Test TearDown" << std::endl;
  }
};
// Base_Pass_Case
TEST_F(Conv3DBackpropFilterProtoTest, Base_Pass_Case){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_Desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = dims_input_size[0] * dims_input_size[1]
      * dims_input_size[2] * dims_input_size[3] * dims_input_size[4];

    tensor_Desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_Desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// No_Filter_Size_Failed
TEST_F(Conv3DBackpropFilterProtoTest, No_Filter_Size_Failed){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

