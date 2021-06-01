#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"


class WtsARQ : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "WtsARQ Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "WtsARQ Proto Test TearDown" << std::endl;
    }
};

TEST_F(WtsARQ, test_wts_arq_infershape_float16)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> y_shape = {16, 6, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);
}

TEST_F(WtsARQ, test_wts_arq_infershape_float)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {16, 6, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_min_shape)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_max_shape)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_max_dims)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 5}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_min_dims)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 2}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 2}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_min_type)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {16, 6, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);
}

TEST_F(WtsARQ, test_wts_arq_infershape_error_w_max_type)
{
    ge::op::WtsARQ op;
    op.UpdateInputDesc("w", create_desc({16, 6, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_min", create_desc({16, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("w_max", create_desc({16, 1, 1, 1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto y = op.GetOutputDescByName("y");
    EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> y_shape = {16, 6, 5, 5};
    EXPECT_EQ(y.GetShape().GetDims(), y_shape);
}
