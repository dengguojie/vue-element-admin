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

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_pool, avg_pool_infershape_test_fp16_2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 128, 32, 32};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic hw SAME NHWC
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({1, -1, -1, 128},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NHWC,
                                                {1, -1, -1, 128},
                                                ge::FORMAT_NHWC,
                                                {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, -1, -1, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic nhw VALID NCHW range -1
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({-1, 128, -1, -1},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NCHW,
                                                {-1, 128, -1, -1},
                                                ge::FORMAT_NCHW,
                                                {{1, -1}, {128, 128}, {20, -1}, {20, 100}}));
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, 128, -1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic -2
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_3) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic c SAME NHWC
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_4) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({1, 24, 24, -1},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NHWC,
                                                {1, 24, 24, -1},
                                                ge::FORMAT_NHWC,
                                                {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(avg_pool, avg_pool_verify_test_dataformat) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "HWCN";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_ksize1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {-1,-1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_ksize2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {-1,2,2,1});
    op.SetAttr("strides", {1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{-1,-1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides", {1,1,1,-1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
