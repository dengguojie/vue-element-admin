#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class SparseApplyFtrlDTest : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "sparse_apply_ftrl_d test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sparse_apply_ftrl_d test TearDown" << std::endl;
  }
};

TEST_F(SparseApplyFtrlDTest, sparse_apply_ftrl_d_infershape_diff_test_1) {
  ge::op::SparseApplyFtrlD op;
  op.UpdateInputDesc("var", create_desc_shape_range({7800, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {7800, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  op.UpdateInputDesc("accum", create_desc_shape_range({7800, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {7800, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  op.UpdateInputDesc("linear", create_desc_shape_range({7800, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {7800, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_var_shape = {7800, 80};
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_var_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_var_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_var_shape_range;
  output_var_desc.GetShapeRange(output_var_shape_range);
  EXPECT_EQ(output_var_shape_range, expected_output_var_shape_range);

  auto output_accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(output_accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_accum_shape = {7800, 80};
  EXPECT_EQ(output_accum_desc.GetShape().GetDims(), expected_output_accum_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_accum_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_accum_shape_range;
  output_accum_desc.GetShapeRange(output_accum_shape_range);
  EXPECT_EQ(output_accum_shape_range, expected_output_accum_shape_range);

  auto output_linear_desc = op.GetOutputDesc("linear");
  EXPECT_EQ(output_linear_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_linear_shape = {7800, 80};
  EXPECT_EQ(output_linear_desc.GetShape().GetDims(), expected_output_linear_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_linear_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_linear_shape_range;
  output_linear_desc.GetShapeRange(output_linear_shape_range);
  EXPECT_EQ(output_linear_shape_range, expected_output_linear_shape_range);
}

TEST_F(SparseApplyFtrlDTest, sparse_apply_ftrl_d_infershape_diff_test_2) {
  ge::op::SparseApplyFtrlD op;
  op.UpdateInputDesc("var", create_desc_shape_range({-1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  op.UpdateInputDesc("accum", create_desc_shape_range({-1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  op.UpdateInputDesc("linear", create_desc_shape_range({-1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, 80}, ge::FORMAT_ND, {{7800, 7800},{80, 80}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_var_shape = {-1, 80};
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_var_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_var_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_var_shape_range;
  output_var_desc.GetShapeRange(output_var_shape_range);
  EXPECT_EQ(output_var_shape_range, expected_output_var_shape_range);

  auto output_accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(output_accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_accum_shape = {-1, 80};
  EXPECT_EQ(output_accum_desc.GetShape().GetDims(), expected_output_accum_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_accum_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_accum_shape_range;
  output_accum_desc.GetShapeRange(output_accum_shape_range);
  EXPECT_EQ(output_accum_shape_range, expected_output_accum_shape_range);

  auto output_linear_desc = op.GetOutputDesc("linear");
  EXPECT_EQ(output_linear_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_linear_shape = {-1, 80};
  EXPECT_EQ(output_linear_desc.GetShape().GetDims(), expected_output_linear_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_linear_shape_range = {{7800, 7800},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_linear_shape_range;
  output_linear_desc.GetShapeRange(output_linear_shape_range);
  EXPECT_EQ(output_linear_shape_range, expected_output_linear_shape_range);
}

TEST_F(SparseApplyFtrlDTest, sparse_apply_ftrl_d_infershape_diff_test_3) {
  ge::op::SparseApplyFtrlD op;
  op.UpdateInputDesc("var", create_desc_shape_range({-1, -1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, -1, 80}, ge::FORMAT_ND, {{7800, 7800},{2, 2},{80, 80}}));
  op.UpdateInputDesc("accum", create_desc_shape_range({-1, -1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, -1, 80}, ge::FORMAT_ND, {{7800, 7800},{2, 2},{80, 80}}));
  op.UpdateInputDesc("linear", create_desc_shape_range({-1, -1, 80}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {-1, -1, 80}, ge::FORMAT_ND, {{7800, 7800},{2, 2},{80, 80}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_var_shape = {-1, -1, 80};
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_output_var_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_var_shape_range = {{7800, 7800},{2, 2},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_var_shape_range;
  output_var_desc.GetShapeRange(output_var_shape_range);
  EXPECT_EQ(output_var_shape_range, expected_output_var_shape_range);

  auto output_accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(output_accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_accum_shape = {-1, -1, 80};
  EXPECT_EQ(output_accum_desc.GetShape().GetDims(), expected_output_accum_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_accum_shape_range = {{7800, 7800},{2, 2},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_accum_shape_range;
  output_accum_desc.GetShapeRange(output_accum_shape_range);
  EXPECT_EQ(output_accum_shape_range, expected_output_accum_shape_range);

  auto output_linear_desc = op.GetOutputDesc("linear");
  EXPECT_EQ(output_linear_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_linear_shape = {-1, -1, 80};
  EXPECT_EQ(output_linear_desc.GetShape().GetDims(), expected_output_linear_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_linear_shape_range = {{7800, 7800},{2, 2},{80, 80}};
  std::vector<std::pair<int64_t, int64_t>> output_linear_shape_range;
  output_linear_desc.GetShapeRange(output_linear_shape_range);
  EXPECT_EQ(output_linear_shape_range, expected_output_linear_shape_range);
}
