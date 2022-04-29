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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 16, 6, 6, 6}, ge::FORMAT_NCDHW));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({1, 6, 1, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 16, 6, 6, 6}, ge::FORMAT_NCDHW));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 16, 19, 19, 19}, ge::FORMAT_NCDHW));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {2, 19, 19, 19, 16}, ge::FORMAT_NDHWC));
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

TEST_F(MaxPool3DTest, x_dims_invalid) {
    ge::op::MaxPool3D op;
    op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {19, 19, 19, 16}, ge::FORMAT_NDHWC));
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

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_001) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_002) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_003) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_004) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_005) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_006) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "ND");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_007) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5};
  std::vector<int32_t> stridesList = {3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_008) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_009) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5};
  std::vector<int32_t> stridesList = {3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_010) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_011) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  std::vector<int32_t> padsList = {2, 2, 2, 2, 2};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_012) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  std::vector<int32_t> padsList = {2, 2, 2, 2, 2, 2};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_013) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  std::vector<int32_t> padsList = {2, 2, 2, 2, 2, 2};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("ceil_mode", padsList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_014) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, 19, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                          {2, 19, 1, 19, 19}, ge::FORMAT_NCDHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  std::vector<int32_t> padsList = {2, 2, 2, 2, 2, 2};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "CALCULATED");
  op.SetAttr("ceil_mode", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 19, 1, 7, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic) {
  ge::op::MaxPool3D op;
  op.UpdateInputDesc("x", create_desc_with_original_shape({2, -1, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                          {2, -1, 1, 19, 19}, ge::FORMAT_NCDHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3, 3};
  std::vector<int32_t> padsList = {2, 2, 2, 2, 2, 2};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "CALCULATED");
  op.SetAttr("ceil_mode", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{2, 2}, {2, 3}, {1, 1}, {19, 19}, {19, 19}};
  op.UpdateInputDesc("x", create_desc_shape_range({2, -1, 1, 19, 19}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {2, -1, 1, 19, 19}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "CALCULATED");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {2, -1, 1, 6, 6};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  //EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{2, 2}, {2, 3}, {1, 1}, {6, 6}, {6, 6}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange1) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {-1, -1, -1, -1, -1}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {1, 1, 1, 1, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 2, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {-1, -1, 0, 0, -1};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {67, 67}, {0, 0}, {0, 0}, {0, 0}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}


TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange2) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {-1, -1, -1, -1, -1}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {4, 4, 4};
  std::vector<int32_t> stridesList = {1, 1, 1, 1, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {-1, -1, -4, -4, -4};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}, {-4, -4}, {-4, -4}, {-4, -4}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange3) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {-1, -1, -1, -1, -1}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {4};
  std::vector<int32_t> stridesList = {1, 1, 1, 1, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {-1, -1, -4, -4, -4};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  for (int i=0;i<actual.size();i++){
      std::cout<<actual[i]<<std::endl;
  }

  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}, {-4, -4}, {-4, -4}, {-4, -4}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange4) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, 150, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {1, -1, 150, 16, 1}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {4};
  std::vector<int32_t> stridesList = {1, 1, 1, 1, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {-2};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 1}, {133, 133}, {147, 147}, {13, 13}, {-2, -2}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange5) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, 150, 16, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {1, -1, 150, 16, 3}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {4};
  std::vector<int32_t> stridesList = {1, 2, 2, 2, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {1, -1, 74, 7, 0};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 1}, {67, 67}, {74, 74}, {7, 7}, {0, 0}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPool3DTest, InfershapeMaxPool3D_dynamic_shaperange6) {
  ge::op::MaxPool3D op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, 150, 16, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                  {-1, -1, 150, 16, 3}, ge::FORMAT_NCDHW, range_x1));
  std::vector<int32_t> ksizeList = {4};
  std::vector<int32_t> stridesList = {1, 2, 2, 2, 1};
  std::vector<int32_t> padsList = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("pads", padsList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("ceil_mode", 0);
  op.SetAttr("dilation", dilation);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected = {-1, -1, 74, 7, 3};
  std::vector<int64_t> actual = output_desc.GetShape().GetDims();
  EXPECT_EQ(expected, actual);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {65, 65}, {74, 74}, {7, 7}, {3, 3}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}
