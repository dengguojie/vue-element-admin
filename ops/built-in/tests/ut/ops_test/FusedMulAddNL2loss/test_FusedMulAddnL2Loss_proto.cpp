#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fused_mul_add_n_l2_loss : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fused_mul_add_n_l2_loss SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fused_mul_add_n_l2_loss TearDown" << std::endl;
    }
};

TEST_F(fused_mul_add_n_l2_loss, fused_mul_add_n_l2_loss_case) {
    ge::op::FusedMulAddNL2loss op;

    op.UpdateInputDesc("x1", create_desc({2, 4, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("x2", create_desc({2, 4, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("x3", create_desc({1,}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y1");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {2, 4, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fused_mul_add_n_l2_loss, VerifyFusedMulAddNL2loss_001) {
  ge::op::FusedMulAddNL2loss op;

  op.UpdateInputDesc("x1", create_desc({2, 4, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({2, 4, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("x3", create_desc({1}, ge::DT_INT64));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(fused_mul_add_n_l2_loss, VerifyFusedMulAddNL2loss_002) {
  ge::op::FusedMulAddNL2loss op;
  op.UpdateInputDesc("x1", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x2", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x3", create_desc({1}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}