#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"


class avg_pool : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool TearDown" << std::endl;
    }
};

TEST_F(avg_pool, avg_pool_infershape_test_fp16_1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides", {1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_pool, avg_pool_infershape_test_fp16_2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 32, 32, 128}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 32, 32, 128}, ge::FORMAT_NHWC));

    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 32, 32, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_pool, avg_pool_infershape_test_fp16_3) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 10, 10, 128}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, 10, 10, 128}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {4, 9, 9, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_pool, avg_pool_infershape_test_fp16_4) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({20, 7, 68, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {20, 7, 68, 3}, ge::FORMAT_NHWC));

    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {20, 7, 68, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
