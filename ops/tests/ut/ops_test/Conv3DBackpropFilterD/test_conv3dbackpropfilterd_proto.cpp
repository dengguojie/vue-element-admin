#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


class Conv3DBackpropFilterDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropFilterD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropFilterD Proto Test TearDown" << std::endl;
  }
};

// Base pass test case 
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Base){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // auto output_desc = op.GetOutputDesc("y");
    // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

// Filter_Size_Error_Failed
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Filter_Size_Error_Failed){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
// Not_Filter_Failed
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Not_Filter_Failed){
    ge::op::Conv3DBackpropFilterD op;
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
