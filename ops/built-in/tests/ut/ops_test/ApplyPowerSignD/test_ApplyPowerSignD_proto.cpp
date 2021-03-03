#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyPowerSignD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyPowerSignD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyPowerSignD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyPowerSignD,
       apply_power_sign_d_infershape_verify_test) {
  ge::op::ApplyPowerSignD op;
  op.UpdateInputDesc("var", create_desc({-1, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({-1, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("logbase", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("sign_decay", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({-1, 48, 16, 32}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {-1, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto m_desc = op.GetOutputDesc("m");
  EXPECT_EQ(m_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_m_output_shape = {-1, 48, 16, 32};
  EXPECT_EQ(m_desc.GetShape().GetDims(), expected_m_output_shape);
}
