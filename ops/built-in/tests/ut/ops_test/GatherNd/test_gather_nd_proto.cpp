#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class gather_nd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_nd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_nd TearDown" << std::endl;
  }
};

TEST_F(gather_nd, gather_nd_infershape_diff_test_1) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND, {{2,2},{2,2},{2,2}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({2, 1, 1}, ge::DT_INT32, ge::FORMAT_ND, {2, 1, 1}, ge::FORMAT_ND, {{2,2},{1,1},{1,1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, 1, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_2) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND, {{2,2},{2,2},{2,2}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND, {{2,2},{2,2},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_3) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND, {{2,2},{2,2},{2,2}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({2, 2, 3}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 3}, ge::FORMAT_ND, {{2,2},{2,2},{3,3}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_4) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND,{{10,10},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {10, 5, 6, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_5) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{3,5},{1,7},{6,9},{6,9}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND,{{10,10},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_6) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,10},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_7) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1, 2}, ge::DT_INT32, ge::FORMAT_ND, {-1, 2}, ge::FORMAT_ND,{{1,10},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, 5, 6, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,10},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_8) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND, {{3,3},{1,1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_nd, gather_nd_infershape_diff_test_9) {
  ge::op::GatherNd op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND, {{3,3},{2,2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
// TODO fix me run failed
//TEST_F(gather_nd, gather_nd_infershape_diff_test_10) {
//  ge::op::GatherNd op;
//  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
//  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,10},{1,2}}));
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  auto output_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
//  std::vector<int64_t> expected_output_shape = {-2};
//  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
//  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,10},{4,4},{5,5},{6,6},{7,7}};
//  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
//  output_desc.GetShapeRange(output_shape_range);
//  EXPECT_EQ(output_shape_range, expected_output_shape_range);
//}
