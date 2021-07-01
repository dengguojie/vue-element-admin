#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "image_ops.h"

class UpsampleNearest1dGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UpsampleNearest1dGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UpsampleNearest1dGradTest TearDown" << std::endl;
  }
};

TEST_F(UpsampleNearest1dGradTest, UpsampleNearest3dGrad_infer_test1_failed) {
  ge::op::UpsampleNearest1dGrad op;
  op.UpdateInputDesc("grad_output",
                     create_desc_with_ori({1,1,10}, ge::DT_INT8, ge::FORMAT_ND, {1,1,10}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dGradTest, UpsampleNearest3dGrad_infer_test2_failed) {
  ge::op::UpsampleNearest1dGrad op;
  op.UpdateInputDesc("grad_output",
                     create_desc_with_ori({1,1,5,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5,5}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dGradTest, UpsampleNearest3dGrad_infer_test3_failed) {
  ge::op::UpsampleNearest1dGrad op;
  op.UpdateInputDesc("grad_output",
                     create_desc_with_ori({1,1,10}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,10}, ge::FORMAT_ND));
  std::vector<int64_t> input_size = {1,1};
  op.SetAttr("input_size", input_size);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dGradTest, UpsampleNearest3dGrad_infer_test1_success) {
  ge::op::UpsampleNearest1dGrad op;
  op.UpdateInputDesc("grad_output",
                     create_desc_with_ori({1,1,10}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,10}, ge::FORMAT_ND));
  std::vector<int64_t> input_size = {1,1,5};
  op.SetAttr("input_size", input_size);
  std::vector<int64_t> output_size = {10};
  op.SetAttr("output_size", output_size);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {1,1,5};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(UpsampleNearest1dGradTest, UpsampleNearest3dGrad_infer_test2_success) {
  ge::op::UpsampleNearest1dGrad op;
  op.UpdateInputDesc("grad_output",
                     create_desc_with_ori({1,1,10}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,10}, ge::FORMAT_ND));
  std::vector<int64_t> input_size = {1,1,5};
  op.SetAttr("input_size", input_size);
  std::vector<float> scales = {2.0};
  op.SetAttr("scales", scales);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {1,1,5};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}
