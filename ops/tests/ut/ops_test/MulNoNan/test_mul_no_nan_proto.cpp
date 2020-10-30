#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class mul_no_nan : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mul_no_nan Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mul_no_nan Proto Test TearDown" << std::endl;
  }
};

TEST_F(mul_no_nan, mul_no_nan_infershape_diff_test){
  ge::op::MulNoNan op;
  op.UpdateInputDesc("x1", create_desc({16, 32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({16, 32}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(mul_no_nan, mul_no_nan_infershape_same_test){
  ge::op::MulNoNan op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

