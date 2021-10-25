#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"


class KlDivLossGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "KlDivLossGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "KlDivLossGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(KlDivLossGradTest, kl_div_loss_grad_infershape_test) {

  ge::op::KlDivLossGrad op;

  op.UpdateInputDesc("grad", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("input", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "sum");

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_var_output_shape = {16, 2, 16, 16};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(KlDivLossGradTest, kl_div_loss_grad_verify_success_test_01) {

  ge::op::KlDivLossGrad op;

  op.UpdateInputDesc("grad", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("input", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "batchmean");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
