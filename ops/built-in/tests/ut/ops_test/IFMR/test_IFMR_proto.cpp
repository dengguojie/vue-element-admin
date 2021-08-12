#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"

class IFMR : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "IFMR Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "IFMR Proto Test TearDown" << std::endl;
    }
};

TEST_F(IFMR, test_ifmr_infershape_float16)
{
    ge::op::IFMR op;
    op.UpdateInputDesc("data", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("data_min", create_desc({1,}, ge::DT_FLOAT16));
    op.UpdateInputDesc("data_max", create_desc({1,}, ge::DT_FLOAT16));
    op.UpdateInputDesc("cumsum", create_desc({512,}, ge::DT_INT32));
    op.SetAttr("min_percentile", 0.999999f);
    op.SetAttr("max_percentile", 0.999999f);
    op.SetAttr("search_range", {7, 13});
    op.SetAttr("search_step", 0.01f);
    op.SetAttr("with_offset", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto scale_desc = op.GetOutputDesc("scale");
    EXPECT_EQ(scale_desc.GetDataType(), ge::DT_FLOAT);
    auto offset_desc = op.GetOutputDesc("offset");
    EXPECT_EQ(offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_scale_shape = {1,};
    EXPECT_EQ(scale_desc.GetShape().GetDims(), expected_scale_shape);
    std::vector<int64_t> expected_offset_shape = {1,};
    EXPECT_EQ(offset_desc.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(IFMR, test_ifmr_infershape_float32)
{
    ge::op::IFMR op;
    op.UpdateInputDesc("data", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_min", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_max", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("cumsum", create_desc({512,}, ge::DT_INT32));
    op.SetAttr("min_percentile", 0.999999f);
    op.SetAttr("max_percentile", 0.999999f);
    op.SetAttr("search_range", {7, 13});
    op.SetAttr("search_step", 0.01f);
    op.SetAttr("with_offset", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto scale_desc = op.GetOutputDesc("scale");
    EXPECT_EQ(scale_desc.GetDataType(), ge::DT_FLOAT);
    auto offset_desc = op.GetOutputDesc("offset");
    EXPECT_EQ(offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_scale_shape = {1,};
    EXPECT_EQ(scale_desc.GetShape().GetDims(), expected_scale_shape);
    std::vector<int64_t> expected_offset_shape = {1,};
    EXPECT_EQ(offset_desc.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(IFMR, test_ifmr_infershape_data_min_float16)
{
    ge::op::IFMR op;
    op.UpdateInputDesc("data", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_min", create_desc({1,}, ge::DT_FLOAT16));
    op.UpdateInputDesc("data_max", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("cumsum", create_desc({512,}, ge::DT_INT32));
    op.SetAttr("min_percentile", 0.999999f);
    op.SetAttr("max_percentile", 0.999999f);
    op.SetAttr("search_range", {7, 13});
    op.SetAttr("search_step", 0.01f);
    op.SetAttr("with_offset", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto scale_desc = op.GetOutputDesc("scale");
    EXPECT_EQ(scale_desc.GetDataType(), ge::DT_FLOAT);
    auto offset_desc = op.GetOutputDesc("offset");
    EXPECT_EQ(offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_scale_shape = {1,};
    EXPECT_EQ(scale_desc.GetShape().GetDims(), expected_scale_shape);
    std::vector<int64_t> expected_offset_shape = {1,};
    EXPECT_EQ(offset_desc.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(IFMR, test_ifmr_infershape_data_max_float16)
{
    ge::op::IFMR op;
    op.UpdateInputDesc("data", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_min", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_max", create_desc({1,}, ge::DT_FLOAT16));
    op.UpdateInputDesc("cumsum", create_desc({512,}, ge::DT_INT32));
    op.SetAttr("min_percentile", 0.999999f);
    op.SetAttr("max_percentile", 0.999999f);
    op.SetAttr("search_range", {7, 13});
    op.SetAttr("search_step", 0.01f);
    op.SetAttr("with_offset", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto scale_desc = op.GetOutputDesc("scale");
    EXPECT_EQ(scale_desc.GetDataType(), ge::DT_FLOAT);
    auto offset_desc = op.GetOutputDesc("offset");
    EXPECT_EQ(offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_scale_shape = {1,};
    EXPECT_EQ(scale_desc.GetShape().GetDims(), expected_scale_shape);
    std::vector<int64_t> expected_offset_shape = {1,};
    EXPECT_EQ(offset_desc.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(IFMR, test_ifmr_infershape_cumsum_int16)
{
    ge::op::IFMR op;
    op.UpdateInputDesc("data", create_desc({32, 3, 5, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_min", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("data_max", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("cumsum", create_desc({512,}, ge::DT_INT16));
    op.SetAttr("min_percentile", 0.999999f);
    op.SetAttr("max_percentile", 0.999999f);
    op.SetAttr("search_range", {7, 13});
    op.SetAttr("search_step", 0.01f);
    op.SetAttr("with_offset", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto scale_desc = op.GetOutputDesc("scale");
    EXPECT_EQ(scale_desc.GetDataType(), ge::DT_FLOAT);
    auto offset_desc = op.GetOutputDesc("offset");
    EXPECT_EQ(offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_scale_shape = {1,};
    EXPECT_EQ(scale_desc.GetShape().GetDims(), expected_scale_shape);
    std::vector<int64_t> expected_offset_shape = {1,};
    EXPECT_EQ(offset_desc.GetShape().GetDims(), expected_offset_shape);
}
