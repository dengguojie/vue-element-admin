#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------MaxPool3D--------------
class MaxPool3DTest: public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MaxPool3D SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPool3D TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DTest, kernel_eq_stride) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "SAME";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("pads", pads);
    op.SetAttr("dilation", dilation);
    op.SetAttr("ceil_mode", 0);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 3, 1, 3, 3, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPool3DTest, kernel_gt_stride) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {1,1,1};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "SAME";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("pads", pads);
    op.SetAttr("dilation", dilation);
    op.SetAttr("ceil_mode", 0);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 6, 1, 6, 6, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPool3DTest, kernel_lt_stride) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "SAME";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("pads", pads);
    op.SetAttr("dilation", dilation);
    op.SetAttr("ceil_mode", 0);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 2, 1, 2, 2, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

