#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyCenteredRMSPropD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyCenteredRMSPropD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyCenteredRMSPropD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyCenteredRMSPropD,
       apply_centered_rms_prop_d_infershape_verify_test) {
  ge::op::ApplyCenteredRMSPropD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("mg", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("ms", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("mom", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto mg_desc = op.GetOutputDesc("mg");
  EXPECT_EQ(mg_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_mg_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(mg_desc.GetShape().GetDims(), expected_mg_output_shape);

  auto ms_desc = op.GetOutputDesc("ms");
  EXPECT_EQ(ms_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_ms_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(ms_desc.GetShape().GetDims(), expected_ms_output_shape);

  auto mom_desc = op.GetOutputDesc("mom");
  EXPECT_EQ(mom_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_mom_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(mom_desc.GetShape().GetDims(), expected_mom_output_shape);
}
