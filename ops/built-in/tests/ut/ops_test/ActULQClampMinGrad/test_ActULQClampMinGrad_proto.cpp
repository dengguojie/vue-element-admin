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
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_BOOL));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output = op.GetOutputDesc("clamp_min_grad");
    EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ActULQClampMinGrad, test_act_ulq_clamp_min_grad_float32)
{
    ge::op::ActULQClampMinGrad op;
    op.UpdateInputDesc("y_grad", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min_mask", create_desc({32, 3, 5, 5}, ge::DT_BOOL));
    op.UpdateInputDesc("x_clamped_loss", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output = op.GetOutputDesc("clamp_min_grad");
    EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output.GetShape().GetDims(), expected_output_shape);
}
