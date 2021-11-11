#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class confusionTranspose_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "confusionTranspose_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "confusionTranspose_test TearDown" << std::endl;
  }
};
TEST_F(confusionTranspose_test, confusionTranspose_test_1) {
  ge::op::ConfusionTranspose op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto input_shape_shape = vector<int64_t>({2, 2, 2, 2});
  std::vector<std::pair<int64_t,int64_t>> input_shape_range = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  auto input_shape_desc = create_desc_shape_range(input_shape_shape, ge::DT_INT32, test_format,
  input_shape_shape, test_format, input_shape_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.UpdateInputDesc("shape", input_shape_desc);
  op.set_attr_perm({3,3,3,3});
  op.set_attr_transpose_first(true);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}
TEST_F(confusionTranspose_test, confusionTranspose_test_2) {
  ge::op::ConfusionTranspose op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto input_shape_shape = vector<int64_t>({2, 2, 2, 2});
  std::vector<std::pair<int64_t,int64_t>> input_shape_range = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  auto input_shape_desc = create_desc_shape_range(input_shape_shape, ge::DT_INT32, test_format,
  input_shape_shape, test_format, input_shape_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.UpdateInputDesc("shape", input_shape_desc);
  op.set_attr_perm({3,3,3,3});
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(confusionTranspose_test, confusionTranspose_test_3) {
  ge::op::ConfusionTranspose op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto input_shape_shape = vector<int64_t>({2, 2, 2, 2});
  std::vector<std::pair<int64_t,int64_t>> input_shape_range = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  auto input_shape_desc = create_desc_shape_range(input_shape_shape, ge::DT_INT32, test_format,
  input_shape_shape, test_format, input_shape_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.UpdateInputDesc("shape", input_shape_desc);
  op.set_attr_transpose_first(true);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(confusionTranspose_test, confusionTranspose_test_4) {
  ge::op::ConfusionTranspose op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto input_shape_shape = vector<int64_t>({2, 2, 2, 2});
  std::vector<std::pair<int64_t,int64_t>> input_shape_range = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  auto input_shape_desc = create_desc_shape_range(input_shape_shape, ge::DT_INT32, test_format,
  input_shape_shape, test_format, input_shape_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.UpdateInputDesc("shape", input_shape_desc);
  op.set_attr_perm({3,3,3,3});
  op.set_attr_transpose_first(false);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}