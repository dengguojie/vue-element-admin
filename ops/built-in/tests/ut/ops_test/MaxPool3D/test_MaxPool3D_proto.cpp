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

TEST_F(MaxPool3DTest, kernel_eq_stride_same_ndhwc) {
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
    std::vector<int64_t> expected = {1, 3, 3, 3, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_eq_stride_valid_ndhwc) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "VALID";
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
    std::vector<int64_t> expected = {1, 3, 3, 3, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_gt_stride_same_ndhwc) {
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
    std::vector<int64_t> expected = {1, 6, 6, 6, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_gt_stride_valid_ndhwc) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {1,1,1};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "VALID";
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
    std::vector<int64_t> expected = {1, 5, 5, 5, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_same_ndhwc) {
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
    std::vector<int64_t> expected = {1, 2, 2, 2, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_same_ncdhw) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 16, 6, 6, 6}, ge::FORMAT_NCDHW));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "SAME";
    std::string data_format= "NCDHW";

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
    std::vector<int64_t> expected = {1, 16, 2, 2, 2};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    for(int i = 0; i < actual.size(); i++) {
        std::cout<<actual[i]<<std::endl;
    }
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_valid_ncdhw) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 16, 6, 6, 6}, ge::FORMAT_NCDHW));
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {0,0,0,0,0,0};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "VALID";
    std::string data_format= "NCDHW";

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
    std::vector<int64_t> expected = {1, 16, 2, 2, 2};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    for(int i = 0; i < actual.size(); i++) {
        std::cout<<actual[i]<<std::endl;
    }
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_calculated_ncdhw) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 16, 19, 19, 19}, ge::FORMAT_NCDHW));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {2,2,2,2,2,2};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
    std::string data_format= "NCDHW";

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
    std::vector<int64_t> expected = {2, 16, 7, 7, 7};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    for(int i = 0; i < actual.size(); i++) {
        std::cout<<actual[i]<<std::endl;
    }
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_calculated_ndhwc_ceil_mode_1) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {2,2,2,2,2,2};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("pads", pads);
    op.SetAttr("dilation", dilation);
    op.SetAttr("ceil_mode", 1);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected = {2, 7, 7, 7, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, kernel_lt_stride_calculated_ndhwc_ceil_mode_0) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {1,1,1,1,1,1};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
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
    std::vector<int64_t> expected = {2, 6, 6, 6, 16};
    std::vector<int64_t> actual = output_desc.GetShape().GetDims();
    EXPECT_EQ(expected.size(), actual.size());
    for(int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST_F(MaxPool3DTest, pads_absent) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {2,2,2,2,2,2};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("dilation", dilation);
    op.SetAttr("ceil_mode", 0);
    auto ret = op.InferShapeAndType();
    //EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, ceil_mode_invalid) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {2,2,2,2,2,2};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("padding", padding);
    op.SetAttr("dilation", dilation);
    op.SetAttr("pads", pads);
    op.SetAttr("ceil_mode", 999);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, padding_absent) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
    std::vector<int64_t> ksize = {5,5,5};
    std::vector<int64_t> strides = {3,3,3};
    std::vector<int64_t> pads = {2,2,2,2,2,2};
    std::vector<int64_t> dilation = {1,1,1,1,1,1};
    std::string padding = "CALCULATED";
    std::string data_format= "NDHWC";

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("dilation", dilation);
    op.SetAttr("pads", pads);
    op.SetAttr("ceil_mode", 0);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

