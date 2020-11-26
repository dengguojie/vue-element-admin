#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class fused_mul_apply_momentum_extern : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fused_mul_apply_momentum_extern SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fused_mul_apply_momentum_extern TearDown" << std::endl;
    }
};

TEST_F(fused_mul_apply_momentum_extern, fused_mul_apply_momentum_extern_case) {
    ge::op::FusedMulApplyMomentumExtern op;

    op.UpdateInputDesc("var", create_desc({144,16,16,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("accumj", create_desc({144,16,16,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("x1", create_desc({144,16,16,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("momentum", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("x2", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("var_copy", create_desc({144, 16, 16, 16}, ge::DT_FLOAT16));

    op.SetAttr("use_nesterov", false);
    op.SetAttr("use_locking", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("var");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {144,16,16,16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}