#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------MaxPool3DGradGradD--------------
class MaxPool3DGradGradDTest: public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MaxPool3DGradGradD SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPool3DGradGradD TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DGradGradDTest, kernel_eq_stride) {
    ge::op::MaxPool3DGradGrad op;
    op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 3, 3, 3, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 3, 3, 3, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 6, 6, 6, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPool3DGradGradDTest, kernel_gt_stride) {
    ge::op::MaxPool3DGradGrad op;
    op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 3, 3, 3, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 3, 3, 3, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {3,3,3};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 6, 6, 6, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(MaxPool3DGradGradDTest, kernel_lt_stride) {
    ge::op::MaxPool3DGradGrad op;
    op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 6, 6, 6, 16};
    //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
