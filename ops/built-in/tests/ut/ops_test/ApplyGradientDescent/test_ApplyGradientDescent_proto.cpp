#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyGradientDescent : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyGradientDescent Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyGradientDescent Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyGradientDescent,
       apply_gradient_descent_infershape_verify_test) {
  ge::op::ApplyGradientDescent op;
  op.UpdateInputDesc("var", create_desc({-1, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("alpha", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("delta", create_desc({-1, 48, 16, 32}, ge::DT_FLOAT));


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {-1, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

}