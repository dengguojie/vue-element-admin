#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class gather : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather TearDown" << std::endl;
  }
};

TEST_F(gather, gather_infershape_diff_test_1) {
  ge::op::Gather op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3,-1,-1,5,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{1,10},{4,4},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
