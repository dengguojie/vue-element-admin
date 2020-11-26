#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyAdamWithAmsgradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamWithAmsgrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamWithAmsgrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAdamWithAmsgradProtoTest,
       apply_adam_with_amsgrad_infershape_verify_test) {
  ge::op::ApplyAdamWithAmsgrad op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(ApplyAdamWithAmsgradProtoTest,
       apply_adam_with_amsgrad_verify_fail_test) {
  ge::op::ApplyAdamWithAmsgrad op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}


class ApplyAdamWithAmsgradDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamWithAmsgradD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamWithAmsgradD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAdamWithAmsgradDProtoTest,
       apply_adam_with_amsgrad_d_infershape_verify_test) {
  ge::op::ApplyAdamWithAmsgradD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("beta1", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto m_desc = op.GetOutputDesc("m");
  EXPECT_EQ(m_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_m_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(m_desc.GetShape().GetDims(), expected_m_output_shape);

  auto v_desc = op.GetOutputDesc("v");
  EXPECT_EQ(v_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_v_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(v_desc.GetShape().GetDims(), expected_v_output_shape);

  auto vhat_desc = op.GetOutputDesc("vhat");
  EXPECT_EQ(vhat_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_vhat_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(vhat_desc.GetShape().GetDims(), expected_vhat_output_shape);
}

TEST_F(ApplyAdamWithAmsgradDProtoTest,
       apply_adam_with_amsgrad_d_verify_fail0_test) {
  ge::op::ApplyAdamWithAmsgradD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("beta1", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ApplyAdamWithAmsgradDProtoTest,
       apply_adam_with_amsgrad_d_verify_fail1_test) {
  ge::op::ApplyAdamWithAmsgradD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("bet", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
