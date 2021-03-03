#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class mvn_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mvn_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mvn_v2 TearDown" << std::endl;
  }
};

TEST_F(mvn_v2, mvn_v2_infershape_diff_test_1) {
  ge::op::MVNV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {3, 4, 5, 6}, ge::FORMAT_NCHW, {{3,3},{4,4},{5,5},{6,6}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 4, 5, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(mvn_v2, mvn_v2_infershape_diff_test_2) {
  ge::op::MVNV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6}, ge::DT_FLOAT, ge::FORMAT_NCHW, {3, 4, 5, 6}, ge::FORMAT_NCHW, {{3,3},{4,4},{5,5},{6,6}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {3, 4, 5, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(mvn_v2, mvn_v2_infershape_diff_test_3) {
  ge::op::MVNV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_NCHW, {3, 4, 5, 6}, ge::FORMAT_NCHW, {{1,3},{1,4},{1,5},{1,6}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
