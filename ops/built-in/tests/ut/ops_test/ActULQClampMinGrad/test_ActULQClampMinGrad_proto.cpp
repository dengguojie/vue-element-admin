#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"


class ActULQClampMinGrad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ActULQClampMinGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ActULQClampMinGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(ActULQClampMinGrad, test_act_ulq_clamp_min_grad_float16)
{
    ge::op::ActULQClampMinGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto clamp_min_grad = op.GetOutputDescByName("clamp_min_grad");
    EXPECT_EQ(clamp_min_grad.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> clamp_min_grad_shape = {};
    EXPECT_EQ(clamp_min_grad.GetShape().GetDims(), clamp_min_grad_shape);
}

TEST_F(ActULQClampMinGrad, test_act_ulq_clamp_min_grad_float)
{
    ge::op::ActULQClampMinGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto clamp_min_grad = op.GetOutputDescByName("clamp_min_grad");
    EXPECT_EQ(clamp_min_grad.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> clamp_min_grad_shape = {};
    EXPECT_EQ(clamp_min_grad.GetShape().GetDims(), clamp_min_grad_shape);
}

TEST_F(ActULQClampMinGrad, test_act_ulq_clamp_min_grad_error_clamp_min_mask_shape)
{
    ge::op::ActULQClampMinGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 5, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto clamp_min_grad = op.GetOutputDescByName("clamp_min_grad");
    EXPECT_EQ(clamp_min_grad.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> clamp_min_grad_shape = {};
    EXPECT_EQ(clamp_min_grad.GetShape().GetDims(), clamp_min_grad_shape);
}

TEST_F(ActULQClampMinGrad, test_act_ulq_clamp_min_grad_error_x_clamped_loss_shape)
{
    ge::op::ActULQClampMinGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 5, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto clamp_min_grad = op.GetOutputDescByName("clamp_min_grad");
    EXPECT_EQ(clamp_min_grad.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> clamp_min_grad_shape = {};
    EXPECT_EQ(clamp_min_grad.GetShape().GetDims(), clamp_min_grad_shape);
}
