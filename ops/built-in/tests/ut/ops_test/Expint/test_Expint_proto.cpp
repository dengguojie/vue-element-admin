#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class expint : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "expint Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "expint Proto Test TearDown" << std::endl;
  }
};

TEST_F(expint, expint_infershape_diff_test){
  ge::op::Expint op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(expint, expint_infershape_same_test){
  ge::op::Expint op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}