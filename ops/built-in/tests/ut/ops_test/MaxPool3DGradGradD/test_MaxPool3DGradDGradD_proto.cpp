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

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_001) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                    ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_002) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));
  op.SetAttr("ksize", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_003) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_004) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_005) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, VerifyMaxPool3DGradGrad_006) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_001) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_002) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc(
      "orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 6, 6, 6, 16}, ge::FORMAT_ND));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_003) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  std::vector<int32_t> strideList = {3, 3, 3};
  op.SetAttr("strides", strideList);
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_004) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  std::vector<int32_t> strideList = {-1};
  std::vector<int32_t> ksizeList = {-1};
  op.SetAttr("strides", strideList);
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_005) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  std::vector<int32_t> strideList = {1, 2, -1, -2, -3};
  std::vector<int32_t> ksizeList = {1, 2, -1, -2, -3};
  op.SetAttr("strides", strideList);
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_006) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  std::vector<int32_t> strideList = {2, -1, -2, -3};
  std::vector<int32_t> ksizeList = {2, -1, -2, -3};
  op.SetAttr("strides", strideList);
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradGradDTest, InfershapeMaxPool3DGradGrad_007) {
  ge::op::MaxPool3DGradGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 6, 6, 6, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 2, 16}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("grads", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 6, 6, 16},
                                                   ge::FORMAT_NCDHW));
  std::vector<int32_t> strideList = {2};
  std::vector<int32_t> ksizeList = {2};
  op.SetAttr("strides", strideList);
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}