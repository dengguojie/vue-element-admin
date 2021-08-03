#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"


class DepthwiseConv2DBackpropInputDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2DBackpropInputD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2DBackpropInputD Proto Test TearDown" << std::endl;
  }
};

// VALID NHWC
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2dBackpropInputDBase1) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 7, 7, 512},
        ge::DT_FLOAT16, ge::FORMAT_NHWC, {128, 7, 7, 512}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 256, 512},
        ge::DT_FLOAT16, ge::FORMAT_HWCN, {1, 1, 256, 512}, ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 14, 14, 256},
        ge::DT_FLOAT16, ge::FORMAT_NHWC, {128, 14, 14, 256}, ge::FORMAT_NHWC));
    op.SetAttr("input_size", {128, 14, 14, 256});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    std::string padding = "VALID";
    op.SetAttr("padding", padding);
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// SAME NCHW
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2dBackpropInputDBase2) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("groups", 1);
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


// check filter type diff x type
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyDtypeTest) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_INT8, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check filter_size out of size 4
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyNoInputSizeTest) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no stride
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyStrideTest1) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyStrideTest2) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyDilationsDimTest) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pad
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyPadsTest1) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyPadsTest2) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyPadsTest3) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// dataformat 
TEST_F(DepthwiseConv2DBackpropInputDProtoTest, DepthwiseConv2DBackpropInputDVerifyDataFormatTest) {
    ge::op::DepthwiseConv2DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 26, 26}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 26, 26},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 26, 26}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "HWCN");
    op.SetAttr("input_size", {1, 16, 26, 26});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
