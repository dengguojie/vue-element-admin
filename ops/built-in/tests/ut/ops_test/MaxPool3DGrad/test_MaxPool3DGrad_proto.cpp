#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include <vector>
#include <string>

using namespace std;
void GetPads(string &model, string &format, vector<int64_t> &in_shape,
        vector<int64_t> &ksize, vector<int64_t> &strides, vector<int64_t> &pads){
    // Only support model is SAME
    int64_t id = (format == "NDHWC") ? in_shape[1] : in_shape[2];
    int64_t ih = (format == "NDHWC") ? in_shape[2] : in_shape[3];
    int64_t iw = (format == "NDHWC") ? in_shape[3] : in_shape[4];

    int64_t kd = (format == "NDHWC") ? ksize[1] : ksize[2];
    int64_t kh = (format == "NDHWC") ? ksize[2] : ksize[3];
    int64_t kw = (format == "NDHWC") ? ksize[3] : ksize[4];

    int64_t sd = (format == "NDHWC") ? strides[1] : strides[2];
    int64_t sh = (format == "NDHWC") ? strides[2] : strides[3];
    int64_t sw = (format == "NDHWC") ? strides[3] : strides[4];

    int64_t out_d = 0;
    int64_t out_h = 0;
    int64_t out_w = 0;

    int64_t pad_h = 0;
    int64_t pad_hw_top = 0;
    int64_t pad_hw_bottom = 0;
    int64_t pad_w = 0;
    int64_t pad_hw_left = 0;
    int64_t pad_hw_right = 0;
    int64_t pad_d = 0;
    int64_t pad_d_top = 0;
    int64_t pad_d_bottom = 0;

    if (model == "SAME"){
        out_d = (id + sd - 1) / sd;
        out_h = (ih + sh - 1) / sh;
        out_w = (iw + sw - 1) / sw;

        pad_h = (out_h - 1) * sh + kh - ih;
        pad_h = (pad_h > 0) ? pad_h : 0;
        pad_hw_top = pad_h / 2;
        pad_hw_bottom = pad_h - pad_hw_top;

        pad_w = (out_w - 1) * sw + kw - iw;
        pad_w = (pad_w > 0) ? pad_w : 0;
        pad_hw_left = pad_w / 2;
        pad_hw_right = pad_w - pad_hw_left;

        pad_d = (out_d - 1) * sd + kd - id;
        pad_d_top = pad_d / 2;
        pad_d_bottom = pad_d - pad_d_top;

        pads[0] = pad_d_top;
        pads[1] = pad_d_bottom;
        pads[2] = pad_hw_top;
        pads[3] = pad_hw_bottom;
        pads[4] = pad_hw_left;
        pads[5] = pad_hw_right;
    }
}


// ----------------MaxPool3DGrad--------------
class max_pool3d_grad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "max_pool3d_grad SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "max_pool3d_grad TearDown" << std::endl;
    }
};

TEST_F(max_pool3d_grad, max_pool3d_grad_infershape_test_0) {
    ge::op::MaxPool3DGrad op;
    op.UpdateInputDesc("orig_x", create_desc_with_ori({16, 32, 7, 13, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {16, 32, 7, 13, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("orig_y", create_desc_with_ori({16, 32, 7, 13, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {16, 32, 7, 13, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("grads", create_desc_with_ori({16, 32, 7, 13, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {16, 32, 7, 13, 16}, ge::FORMAT_NDHWC));
    vector<int64_t> ksize = {1, 6, 6, 6, 1};
    vector<int64_t> strides = {1, 6, 6, 6, 1};
    vector<int64_t> pads = {0, 0, 0, 0, 0, 0};

    string model = "SAME";
    string format = "NDHWC";
    vector<int64_t> in_shape = {16, 32, 7, 13, 16};
    GetPads(model, format, in_shape, ksize, strides, pads);

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {16, 32, 7, 13, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(max_pool3d_grad, max_pool3d_grad_infershape_test_1) {
    ge::op::MaxPool3DGrad op;
    op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
    vector<int64_t> ksize = {1, 2, 2, 2, 1};
    vector<int64_t> strides = {1, 1, 1, 1, 1};
    vector<int64_t> pads = {0, 0, 0, 0, 0, 0};

    string model = "VALID";
    string format = "NDHWC";
    vector<int64_t> in_shape = {17, 63, 7, 2, 16};
    GetPads(model, format, in_shape, ksize, strides, pads);

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {17, 63, 7, 2, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(max_pool3d_grad, max_pool3d_grad_infershape_test_2) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> pads = {0, 0, 0, 0, 0, 0};

  string model = "CALCULATED";
  string format = "NDHWC";
  vector<int64_t> in_shape = {17, 63, 7, 2, 16};
  GetPads(model, format, in_shape, ksize, strides, pads);

  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding", model);
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {17, 63, 7, 2, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_001) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_002) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.SetAttr("ksize", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_003) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  vector<int64_t> ksize = {1, 2, 2, 2};
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_004) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_005) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  vector<int64_t> strides = {1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, VerifyMaxPool3DGrad_006) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("orig_y", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                    {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grads", create_desc_with_ori({17, 62, 6, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                   {17, 62, 6, 1, 16}, ge::FORMAT_NDHWC));
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_001) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCHW));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_002) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_003) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_004) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 0, 0, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_005) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);
  op.SetAttr("padding", "CALCULATED");
  op.SetAttr("pads", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_006) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_007) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);
  op.SetAttr("padding", strides);
  op.SetAttr("pads", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_008) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  vector<int64_t> pads = {0, 0, 0, 0, 0};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);
  op.SetAttr("padding", strides);
  op.SetAttr("pads", pads);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, InfershapeMaxPool3DGrad_009) {
  ge::op::MaxPool3DGrad op;
  op.UpdateInputDesc("orig_x", create_desc_with_ori({17, 63, 7, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {17, 63, 7, 2, 16}, ge::FORMAT_NCDHW));
  vector<int64_t> strides = {1, 1, 1, 1, 1};
  vector<int64_t> ksize = {1, 2, 2, 2, 1};
  vector<int64_t> pads = {0, 0, -1, 0, 0, 0};
  op.SetAttr("strides", strides);
  op.SetAttr("ksize", ksize);
  op.SetAttr("padding", strides);
  op.SetAttr("pads", pads);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(max_pool3d_grad, max_pool3d_grad_infershape_test_001) {
    ge::op::MaxPool3DGrad op;
    std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
    op.UpdateInputDesc("orig_x", create_desc_shape_range({-1, -1, 150, 16, 3}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                         {-1, -1, 150, 16, 3}, ge::FORMAT_NDHWC, range_x1));
    op.UpdateInputDesc("orig_y", create_desc_shape_range({-1, -1, 150, 16, 3}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                         {-1, -1, 150, 16, 3}, ge::FORMAT_NDHWC, range_x1));
    op.UpdateInputDesc("grads", create_desc_shape_range({-1, -1, 150, 16, 3}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                         {-1, -1, 150, 16, 3}, ge::FORMAT_NDHWC, range_x1));
    vector<int64_t> ksize = {1, 2, 2, 2, 1};
    vector<int64_t> strides = {1, 1, 1, 1, 1};
    vector<int64_t> pads = {0, 0, 0, 0, 0, 0};

    string model = "VALID";
    string format = "NDHWC";
    vector<int64_t> in_shape = {17, 63, 7, 2, 16};
    GetPads(model, format, in_shape, ksize, strides, pads);

    op.SetAttr("ksize", ksize);
    op.SetAttr("strides", strides);
    op.SetAttr("pads", pads);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1, -1, 150, 16, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_range;
    std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}, {5, 23}, {13, 25}, {0, 0}};
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_range, expected_range);

}
