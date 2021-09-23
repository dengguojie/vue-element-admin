#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class ScatterAddWithAxisTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterAddWithAxisTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterAddWithAxisTest TearDown" << std::endl;
  }
};

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_infershape_diff_test_1) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND, {{2,2},{2,2},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,2},{2,2},{2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_infershape_diff_test_2) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({2, 2, -1}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, -1}, ge::FORMAT_ND, {{2,2},{2,2},{2,20}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, 2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,2},{2,2},{2,20}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_infershape_diff_test_3) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{2,2},{2,2},{3,5}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,2},{2,2},{3,5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_infershape_diff_test_4) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({1,56,56,64}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {1,56,56,64}, ge::FORMAT_NHWC, {{2,2},{2,2},{3,5}}));
  op.UpdateOutputDesc("var", create_desc_shape_range({}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {}, ge::FORMAT_NHWC, {}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_NC1HWC0);
  std::vector<int64_t> expected_output_shape = {1,4,56,56,16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_verify_success) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({1,56,56,64}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {1,56,56,64}, ge::FORMAT_NHWC, {{2,2},{2,2},{3,5}}));
  op.UpdateInputDesc("updates", create_desc_shape_range({1,56,56,64}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {1,56,56,64}, ge::FORMAT_NHWC, {{2,2},{2,2},{3,5}}));
  op.UpdateOutputDesc("var", create_desc_shape_range({}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {}, ge::FORMAT_NHWC, {}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ScatterAddWithAxisTest, scatter_add_with_axis_verify_failure) {
  ge::op::ScatterAddWithAxis op;
  op.UpdateInputDesc("var", create_desc_shape_range({1,56,56,64}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {1,56,56,64}, ge::FORMAT_NHWC, {{2,2},{2,2},{3,5}}));
  op.UpdateOutputDesc("var", create_desc_shape_range({}, ge::DT_INT32, ge::FORMAT_NC1HWC0, {}, ge::FORMAT_NHWC, {}));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
