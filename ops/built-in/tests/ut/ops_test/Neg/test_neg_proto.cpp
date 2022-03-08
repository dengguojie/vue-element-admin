#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

// ----------------Neg-------------------
class Neg : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Neg SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Neg TearDown" << std::endl;
    }
};

TEST_F(Neg, Neg_infershape_test_0) {
  ge::op::Neg op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 2, 1}, ge::FORMAT_ND, {{2,2},{2,2},{1,1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 2, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(Neg, Neg_infershape_test_1) {
  ge::op::Neg op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4, 5, 6, -1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{3,8}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {3, 4, 5, 6, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{5,5},{6,6},{3,8}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(Neg, Neg_infershape_test_2) {
  ge::op::Neg op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1,2}, ge::DT_INT32, ge::FORMAT_ND, {-1,2}, ge::FORMAT_ND, {{1,8}, {2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,8}, {2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(Neg, Neg_infershape_test_3) {
  ge::op::Neg op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1}, ge::DT_DOUBLE, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,55}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,55}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
