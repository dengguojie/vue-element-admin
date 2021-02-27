#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "avg_pool_1d_ops.h"

class AvgPool1DAvgMatrix : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool1DAvgMatrix SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool1DAvgMatrix TearDown" << std::endl;
  }
};

TEST_F(AvgPool1DAvgMatrix, avg_pool1d_avg_matrix_infershape_test){
  ge::op::AvgPool1DAvgMatrix op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 1, 1,4}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
      {1, 1, 1,4}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 2);
  op.SetAttr("strides", 2);
  op.SetAttr("pads", {1,2});
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1,16,1,3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AvgPool1DAvgMatrix, AvgPool1DAvgMatrix_infershape_zero){
  ge::op::AvgPool1DAvgMatrix op;
  op.UpdateInputDesc("x", create_desc_with_ori(
     {1, 1, 1,4}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
     {1, 1, 1,4}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 2);
  op.SetAttr("strides", 0);
  op.SetAttr("pads", {1,2});
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DAvgMatrix, avg_pool1d_avg_matrix_infershape_test_UNKNOWN_DIM){
  ge::op::AvgPool1DAvgMatrix op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 1, 1,ge::UNKNOWN_DIM}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
      {1, 1, 1,ge::UNKNOWN_DIM}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 2);
  op.SetAttr("strides", 2);
  op.SetAttr("pads", {1,2});
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1,16,1,ge::UNKNOWN_DIM};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}