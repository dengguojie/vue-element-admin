#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

// ----------------NLLLossGrad--------------
class nll_loss_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "nll_loss_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "nll_loss_grad TearDown" << std::endl;
  }
};

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test1) {
  ge::op::NLLLossGrad op;
  op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("x_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test2) {
  ge::op::NLLLossGrad op;
  op.UpdateInputDesc(
      "x", create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 16}, ge::FORMAT_ND, {{1, -1}, {1, -1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("x_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, -1}, {1, -1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test3) {
  ge::op::NLLLossGrad op;
  op.UpdateInputDesc(
      "x", create_desc_shape_range({-1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 5}, ge::FORMAT_ND, {{1, 25}, {5, 5}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("x_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 25}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test4) {
  ge::op::NLLLossGrad op;
  op.UpdateInputDesc(
      "x", create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 5}, ge::FORMAT_ND, {{5, 25}, {1, 5}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("x_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{5, 25}, {1, 5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test5) {
  ge::op::NLLLossGrad op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2, }, ge::DT_FLOAT, ge::FORMAT_ND, {2, 5}, ge::FORMAT_ND, {}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("x_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
