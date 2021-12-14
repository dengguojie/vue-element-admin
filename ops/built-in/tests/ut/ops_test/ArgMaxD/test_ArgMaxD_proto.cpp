#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class arg_max_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "arg_max_d Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "arg_max_d Proto Test TearDown" << std::endl;
  }
};

TEST_F(arg_max_d, arg_max_d_infershape_1){
  // set input info
  auto input_shape = vector<int64_t>({4, 3, 4});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;
  int dimension = 1;

  // expect result
  std::vector<int64_t> expected_shape = {4, 4};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::ArgMaxD op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("dimension", dimension);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(arg_max_d, arg_max_d_infershape_2){
  // set input info
  auto input_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;
  int dimension = 1;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{4, 4}, {4, 4}};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::ArgMaxD op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("dimension", dimension);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(arg_max_d, arg_max_d_infershape_3){
  // set input info
  auto input_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;
  int dimension = 1;

  // expect result
  std::vector<int64_t> expected_shape = {-2};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::ArgMaxD op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("dimension", dimension);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(arg_max_d, InfershapeArgMaxD_001){
  ge::op::ArgMaxD op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT));
  op.SetAttr("dimension", 1);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(arg_max_d, InfershapeArgMaxD_002){
  ge::op::ArgMaxD op;
  op.UpdateInputDesc("x", create_desc({4,3,2}, ge::DT_FLOAT));
  op.SetAttr("dimension", false);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}