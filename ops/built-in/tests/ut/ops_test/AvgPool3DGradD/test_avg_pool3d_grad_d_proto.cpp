#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"


class AvgPool3DGradDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DGradD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DGradD Proto Test TearDown" << std::endl;
  }
};

// Base1 pass test case
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_base1)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {2, 2, 1});
    op.SetAttr("strides", {9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// Base2 pass test case
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_base2)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {2, 2, 1});
    op.SetAttr("strides", {9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NDHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

