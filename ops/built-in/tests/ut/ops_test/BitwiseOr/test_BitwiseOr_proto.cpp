#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class bitwise_or : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "bitwise_or SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bitwise_or TearDown" << std::endl;
  }
};

TEST_F(bitwise_or, bitwise_or_infershape_diff_test){
  ge::op::BitwiseOr op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_INT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_INT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(bitwise_or, bitwise_or_infershape_same_test){
  ge::op::BitwiseOr op;
  op.UpdateInputDesc("x1", create_desc({1, 3, 4}, ge::DT_INT16));  
  op.UpdateInputDesc("x2", create_desc({1, 3, 4}, ge::DT_INT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
