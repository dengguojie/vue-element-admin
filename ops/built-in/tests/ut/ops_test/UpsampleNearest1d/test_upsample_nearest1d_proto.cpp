#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "image_ops.h"

class UpsampleNearest1dTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UpsampleNearest1dTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UpsampleNearest1dTest TearDown" << std::endl;
  }
};

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test1_failed) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5}, ge::DT_INT8, ge::FORMAT_ND, {1,1,5}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test2_failed) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5,5}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test3_failed) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5}, ge::FORMAT_ND));
  std::vector<int64_t> output_size = {10, 10};
  op.SetAttr("output_size", output_size);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test4_failed) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5}, ge::FORMAT_ND));
  std::vector<float> scales = {2.0, 2.0};
  op.SetAttr("scales", scales);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test1_success) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5}, ge::FORMAT_ND));
  std::vector<int64_t> output_size = {10};
  op.SetAttr("output_size", output_size);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {1,1,10};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(UpsampleNearest1dTest, UpsampleNearest1d_infer_test2_success) {
  ge::op::UpsampleNearest1d op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({1,1,5}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,5}, ge::FORMAT_ND));
  std::vector<float> scales = {2.0};
  op.SetAttr("scales", scales);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {1,1,10};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}
