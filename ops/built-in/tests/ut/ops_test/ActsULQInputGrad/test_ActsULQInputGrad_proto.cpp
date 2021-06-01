#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"


class ActsULQInputGrad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ActsULQInputGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ActsULQInputGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(ActsULQInputGrad, test_acts_ulq_input_grad_float16)
{
    ge::op::ActsULQInputGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_max_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto x_grad = op.GetOutputDescByName("x_grad");
    EXPECT_EQ(x_grad.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> x_grad_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_grad.GetShape().GetDims(), x_grad_shape);
}

TEST_F(ActsULQInputGrad, test_acts_ulq_input_grad_float)
{
    ge::op::ActsULQInputGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto x_grad = op.GetOutputDescByName("x_grad");
    EXPECT_EQ(x_grad.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> x_grad_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_grad.GetShape().GetDims(), x_grad_shape);
}

TEST_F(ActsULQInputGrad, test_acts_ulq_input_grad_error_clamp_min_mask_shape)
{
    ge::op::ActsULQInputGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 5, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ActsULQInputGrad, test_acts_ulq_input_grad_error_clamp_max_mask_shape)
{
    ge::op::ActsULQInputGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max_mask", create_desc({32, 5, 5, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ActsULQInputGrad, test_acts_ulq_input_grad_error_clamp_max_mask_type)
{
    ge::op::ActsULQInputGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max_mask", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto x_grad = op.GetOutputDescByName("x_grad");
    EXPECT_EQ(x_grad.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> x_grad_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_grad.GetShape().GetDims(), x_grad_shape);
}
