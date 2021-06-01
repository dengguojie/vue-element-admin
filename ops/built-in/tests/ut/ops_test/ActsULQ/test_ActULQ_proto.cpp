#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"


class ActsULQ : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ActsULQ Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ActsULQ Proto Test TearDown" << std::endl;
    }
};

TEST_F(ActsULQ, test_acts_ulq_float16)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> y_shape = {32, 3, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);

    auto clamp_min_mask = op.GetOutputDescByName("clamp_min_mask");
    std::vector<int64_t> clamp_min_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_min_mask.GetShape().GetDims(), clamp_min_mask_shape);

    auto clamp_max_mask = op.GetOutputDescByName("clamp_max_mask");
    std::vector<int64_t> clamp_max_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_max_mask.GetShape().GetDims(), clamp_max_mask_shape);

    auto x_clamped_loss = op.GetOutputDescByName("x_clamped_loss");
    EXPECT_EQ(x_clamped_loss.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> x_clamped_loss_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_clamped_loss.GetShape().GetDims(), x_clamped_loss_shape);
}

TEST_F(ActsULQ, test_acts_ulq_float)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {32, 3, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);

    auto clamp_min_mask = op.GetOutputDescByName("clamp_min_mask");
    std::vector<int64_t> clamp_min_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_min_mask.GetShape().GetDims(), clamp_min_mask_shape);

    auto clamp_max_mask = op.GetOutputDescByName("clamp_max_mask");
    std::vector<int64_t> clamp_max_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_max_mask.GetShape().GetDims(), clamp_max_mask_shape);

    auto x_clamped_loss = op.GetOutputDescByName("x_clamped_loss");
    EXPECT_EQ(x_clamped_loss.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> x_clamped_loss_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_clamped_loss.GetShape().GetDims(), x_clamped_loss_shape);
}

TEST_F(ActsULQ, test_acts_ulq_error_clamp_min_shape)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min", create_desc({2}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ActsULQ, test_acts_ulq_error_clamp_max_shape)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max", create_desc({2}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ActsULQ, test_acts_ulq_error_clamp_min_type)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {32, 3, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);

    auto clamp_min_mask = op.GetOutputDescByName("clamp_min_mask");
    std::vector<int64_t> clamp_min_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_min_mask.GetShape().GetDims(), clamp_min_mask_shape);

    auto clamp_max_mask = op.GetOutputDescByName("clamp_max_mask");
    std::vector<int64_t> clamp_max_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_max_mask.GetShape().GetDims(), clamp_max_mask_shape);

    auto x_clamped_loss = op.GetOutputDescByName("x_clamped_loss");
    EXPECT_EQ(x_clamped_loss.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> x_clamped_loss_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_clamped_loss.GetShape().GetDims(), x_clamped_loss_shape);
}

TEST_F(ActsULQ, test_acts_ulq_error_clamp_max_type)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {32, 3, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);

    auto clamp_min_mask = op.GetOutputDescByName("clamp_min_mask");
    std::vector<int64_t> clamp_min_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_min_mask.GetShape().GetDims(), clamp_min_mask_shape);

    auto clamp_max_mask = op.GetOutputDescByName("clamp_max_mask");
    std::vector<int64_t> clamp_max_mask_shape = {32, 3, 5, 5};
    EXPECT_EQ(clamp_max_mask.GetShape().GetDims(), clamp_max_mask_shape);

    auto x_clamped_loss = op.GetOutputDescByName("x_clamped_loss");
    EXPECT_EQ(x_clamped_loss.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> x_clamped_loss_shape = {32, 3, 5, 5};
    EXPECT_EQ(x_clamped_loss.GetShape().GetDims(), x_clamped_loss_shape);
}
