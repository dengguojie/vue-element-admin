#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class space_to_depth_infer_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "space_to_depth_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "space_to_depth_infer_test TearDown" << std::endl;
  }
};

TEST_F(space_to_depth_infer_test, space_to_depth_infer_test_1) {

  // set input info
  auto input_shape = vector<int64_t>({12, 18, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{12, 12}, {18, 18}, {20, 20}, {16, 16}};
  auto block_size = 2;
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {12, 9, 10, 64};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::SpaceToDepth op;
  op.UpdateInputDesc("x", input_desc);
  op.set_attr_block_size(block_size);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(space_to_depth_infer_test, space_to_depth_infer_test_4) {

  // set input info
  auto input_shape = vector<int64_t>({12, 16, 18, 20});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{12, 12}, {16, 16}, {18, 18}, {20, 20}};
  auto block_size = 2;
  auto test_format = ge::FORMAT_NCHW;

  // expect result
  std::vector<int64_t> expected_shape = {12, 64, 9, 10};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::SpaceToDepth op;
  op.UpdateInputDesc("x", input_desc);
  op.set_attr_block_size(block_size);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(space_to_depth_infer_test, space_to_depth_infer_test_2) {

  // set input info
  auto input_shape = vector<int64_t>({40, -1, -1, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{40, 40}, {8, 8}, {12, 12}, {16, 16}};
  auto block_size = 2;
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {40, -1, -1, 64};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{40, 40}, {4, 4}, {6, 6}, {64, 64}};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::SpaceToDepth op;
  op.UpdateInputDesc("x", input_desc);
  op.set_attr_block_size(block_size);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(space_to_depth_infer_test, space_to_depth_infer_test_5) {

  // set input info
  auto input_shape = vector<int64_t>({40, 16, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{40, 40}, {16, 16}, {8, 8}, {12, 12}};
  auto block_size = 2;
  auto test_format = ge::FORMAT_NCHW;

  // expect result
  std::vector<int64_t> expected_shape = {40, 64, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{40, 40}, {64, 64}, {4, 4}, {6, 6}};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::SpaceToDepth op;
  op.UpdateInputDesc("x", input_desc);
  op.set_attr_block_size(block_size);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(space_to_depth_infer_test, space_to_depth_infer_test_3) {

  // set input info
  auto input_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {12, 12}, {20, 20}, {16, 16}};
  auto block_size = 2;
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {-2};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::SpaceToDepth op;
  op.UpdateInputDesc("x", input_desc);
  op.set_attr_block_size(block_size);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}