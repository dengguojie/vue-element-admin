#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "graph/utils/tensor_utils.h"

using namespace ge;
using namespace op;

class flattenV2_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "flattenV2_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "flattenV2_test TearDown" << std::endl;
  }
};
TEST_F(flattenV2_test, flattenV2_test_1) {
  ge::op::FlattenV2 op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  std::vector<int64_t> expected_shape = {10, 3520};
  op.UpdateInputDesc("x", input_x_desc);
  auto tensor_x = op.GetInputDescByName("x");
  tensor_x.SetRealDimCnt(4);
  op.UpdateInputDesc("x", tensor_x);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(flattenV2_test, flattenV2_test_2) {
  ge::op::FlattenV2 op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  std::vector<int64_t> expected_shape = {10, 3520};
  op.UpdateInputDesc("x", input_x_desc);
  auto tensor_x = op.GetInputDescByName("x");
  tensor_x.SetRealDimCnt(4);
  op.UpdateInputDesc("x", tensor_x);
  op.set_attr_axis(-3);
  op.set_attr_end_axis(-1);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(flattenV2_test, flattenV2_test_3) {
  ge::op::FlattenV2 op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  op.UpdateInputDesc("x", input_x_desc);
  auto tensor_x = op.GetInputDescByName("x");
  tensor_x.SetRealDimCnt(4);
  op.UpdateInputDesc("x", tensor_x);
  op.set_attr_axis(-6);
  op.set_attr_end_axis(-4);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(flattenV2_test, flattenV2_test_4) {
  ge::op::FlattenV2 op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  op.UpdateInputDesc("x", input_x_desc);
  auto tensor_x = op.GetInputDescByName("x");
  tensor_x.SetRealDimCnt(4);
  op.UpdateInputDesc("x", tensor_x);
  op.set_attr_axis(-3);
  op.set_attr_end_axis(-8);
  // check result
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}