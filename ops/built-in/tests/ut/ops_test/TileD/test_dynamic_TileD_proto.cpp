#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"

class tile_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tile_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tile_d TearDown" << std::endl;
  }
};

TEST_F(tile_d, tile_d_infershape_test_0) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({2, -1}, ge::DT_FLOAT,
ge::FORMAT_ND, {2, -1}, ge::FORMAT_ND, {{2,2}, {1,9}}));
op.SetAttr("multiples",{7,8,9});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
std::vector<int64_t> expected_output_shape = {7, 16, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{7, 7},
{16, 16},{9, 81}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(tile_d, tile_d_infershape_test_1) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({-1, 3}, ge::DT_INT32,
ge::FORMAT_ND, {-1, 3}, ge::FORMAT_ND, {{1,10}, {3, 3}}));
op.SetAttr("multiples",{1, 2, 3, 4, 5});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {1, 2, 3, -1, 15};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 1},
{2, 2},{3, 3}, {4, 40}, {15, 15}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(tile_d, tile_d_infershape_test_2) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32,
ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,10}, {3, 3}}));
op.SetAttr("multiples",{2,3});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,-1},
{3, -1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(tile_d, test_tile_fp32_001) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({-1}, ge::DT_INT32,
ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,38}}));
op.SetAttr("multiples",{2});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,76}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(tile_d, test_tile_fp32_003) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({-1}, ge::DT_INT32,
ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,-1}}));
op.SetAttr("multiples",{334});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{334,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}



TEST_F(tile_d, test_tile_fp16_004) {
ge::op::TileD op;
op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32,
ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,-1}, {1, 20}}));
op.SetAttr("multiples",{2,3});
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,-1}, {3,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}



TEST_F(tile_d, test_tile_fp16_010) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({8, 7, 8}, ge::DT_INT32,
ge::FORMAT_ND, {8, 7, 8}, ge::FORMAT_ND, {{8,8}, {7,7},{8,8}}));
op.UpdateInputDesc("multiples", create_desc_shape_range({-1}, ge::DT_INT32,
ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,-1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1, -1, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1}, {1,-1},{1,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(tile_d, test_tile_fp16_011) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({8, -1, 8}, ge::DT_INT32,
ge::FORMAT_ND, {8, -1, 8}, ge::FORMAT_ND, {{8,8}, {1,9},{8,8}}));
// multi format set the same as input
op.UpdateInputDesc("multiples", create_desc_shape_range({-1}, ge::DT_INT32,
ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,-1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1, -1, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1}, {1,-1},{1,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(tile_d, test_tile_fp32_012) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({38}, ge::DT_INT32,
ge::FORMAT_ND, {38}, ge::FORMAT_ND, {{38,38}}));
op.UpdateInputDesc("multiples", create_desc_shape_range({-2}, ge::DT_INT32,
ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,-1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(tile_d, test_tile_fp16_013) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({8,-1,8}, ge::DT_INT32,
ge::FORMAT_ND, {8,-1,8}, ge::FORMAT_ND, {{8,8}, {1,9},{8,8}}));
op.UpdateInputDesc("multiples", create_desc_shape_range({-2}, ge::DT_INT32,
ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,-1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1, -1, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{1,-1},{1,-1}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}