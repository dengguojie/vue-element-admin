#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

// ----------------NLLLossGrad--------------
class nll_loss_grad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "nll_loss_grad SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "nll_loss_grad TearDown" << std::endl;
    }
};

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test) {
    ge::op::NLLLossGrad op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
