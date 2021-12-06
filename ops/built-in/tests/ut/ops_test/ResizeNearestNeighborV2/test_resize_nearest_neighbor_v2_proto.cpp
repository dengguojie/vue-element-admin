#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;
using namespace ut_util;

class resize_nearest_neighbor_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "resize_nearest_neighbor_v2_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "resize_nearest_neighbor_v2_infer_test TearDown" << std::endl;
  }
};

TEST_F(resize_nearest_neighbor_v2_infer_test, resize_nearest_neighbor_v2_static_shape_nchw) {
  // input x info
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT;
  auto input_x_format = FORMAT_NCHW;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{1, -1}, {2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, 5, 1, 2};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range;

  // gen ResizeNearestNeighborV2 op
  auto test_op = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), input_x_dtype);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(resize_nearest_neighbor_v2_infer_test, resize_nearest_neighbor_v2_static_shape_nhwc) {
  // input x info
  auto input_x_shape = vector<int64_t>({-1, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NHWC;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range = {{1, 22}, {5, 5}, {16, 16}, {16, 16}};
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{1, -1}, {2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {-1, 1, 2, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 22}, {1, 1}, {2, 2}, {16, 16}};

  // gen ResizeNearestNeighborV2 op
  auto test_op = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), input_x_dtype);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
