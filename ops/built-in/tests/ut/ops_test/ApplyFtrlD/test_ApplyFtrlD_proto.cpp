#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyFtrlD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyFtrlD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyFtrlD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyFtrlD,
       apply_ftrl_d_infershape_verify_test) {
  ge::op::ApplyFtrlD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("linear", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("l1", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("l2", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr_power", create_desc({1, }, ge::DT_FLOAT));

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

  auto linear_desc = op.GetOutputDesc("linear");
  EXPECT_EQ(linear_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_linear_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(linear_desc.GetShape().GetDims(), expected_linear_output_shape);
}
