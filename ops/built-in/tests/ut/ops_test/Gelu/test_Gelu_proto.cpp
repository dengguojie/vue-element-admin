#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class gelu : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gelu Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelu Proto Test TearDown" << std::endl;
  }
};

TEST_F(gelu, gelu_infershape_diff_test){
  ge::op::Gelu op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(gelu, gelu_infershape_same_test){
  ge::op::Gelu op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

