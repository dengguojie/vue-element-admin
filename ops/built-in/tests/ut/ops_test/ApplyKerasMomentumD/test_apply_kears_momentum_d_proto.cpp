#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

// ----------------ApplyKearsMomentum-------------------
class ApplyKearsMomentumProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyKearsMomentum Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyKearsMomentum Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyKearsMomentumProtoTest,
       apply_keras_momentum_infershape_verify_test) {
  ge::op::ApplyKerasMomentum op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);
  op.SetAttr("use_nesterov", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(ApplyKearsMomentumProtoTest, apply_keras_momentum_verify_test) {
  ge::op::ApplyKerasMomentum op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);
  op.SetAttr("use_nesterov", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}


// ----------------ApplyKearsMomentumD-------------------
class ApplyKearsMomentumDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyKearsMomentumD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyKearsMomentumD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyKearsMomentumDProtoTest,
       apply_keras_momentum_d_infershape_verify_test) {
  ge::op::ApplyKerasMomentumD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);
  op.SetAttr("use_nesterov", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_accum_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(accum_desc.GetShape().GetDims(), expected_accum_output_shape);
}

TEST_F(ApplyKearsMomentumDProtoTest, apply_keras_momentum_d_verify_fail_test) {
  ge::op::ApplyKerasMomentumD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);
  op.SetAttr("use_nesterov", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
