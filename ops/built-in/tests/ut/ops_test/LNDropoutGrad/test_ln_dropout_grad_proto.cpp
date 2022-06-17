#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class ln_dropout_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ln_dropout_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ln_dropout_grad TearDown" << std::endl;
  }
};

TEST_F(ln_dropout_grad, ln_dropout_grad_infershape_diff_test_1) {
  ge::op::LNDropoutGrad op;
  op.UpdateInputDesc("dy", create_desc({16,512,512}, ge::DT_FLOAT));
  op.UpdateInputDesc("x", create_desc({16,512,512}, ge::DT_FLOAT));
  op.UpdateInputDesc("variance", create_desc({16, 512,1}, ge::DT_FLOAT));
  op.UpdateInputDesc("mean", create_desc({16, 512, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("gamma", create_desc({512}, ge::DT_FLOAT));
  op.UpdateInputDesc("mask", create_desc({16,512,512}, ge::DT_UINT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("pd_x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {16, 512, 512};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

