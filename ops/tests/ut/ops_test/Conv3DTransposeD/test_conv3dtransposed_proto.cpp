#include <gtest/gtest.h>
#include <iostream>
#include "nn_calculation_ops.h"
#include "op_proto_test_util.h"


// ---------------Conv3DTransposeD-------------------
class Conv3DTransposeDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DTransposeD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DTransposeD Proto Test TearDown" << std::endl;
  }
};


// base ut1   FORMAT_NDHWC and FORMAT_DHWCN
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDTest) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut2 FORMAT_NCDHW and FORMAT_NCDHW
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDTest2) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 10, 2, 2, 10}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{2, 10, 2, 2, 10},ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({10, 4, 1, 2, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{10, 4, 1, 2, 3},ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 10, 4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{1, 10, 4, 6, 8},ge::FORMAT_NCDHW));
    op.SetAttr("input_size", {1, 10, 4, 6, 8});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut3 FORMAT_NDHWC and FORMAT_NDHWC
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDTest3) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({10, 1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{10, 1, 2, 3, 4},ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//check fm type diff filter type
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyType) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc({2, 2, 2, 10, 10}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({1, 2, 3, 4, 10}, ge::DT_INT32));
    op.UpdateOutputDesc("y", create_desc({1, 4, 6, 8, 10}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("data_format","NCHW");
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//check filter out of size 5
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyFilterLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc({2, 2, 2, 10, 10}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({1, 2, 3, 10}, ge::DT_FLOAT16));
    op.UpdateOutputDesc("y", create_desc({1, 4, 6, 8, 10}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("data_format","NCHW");
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check x size
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyXLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10, 11}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10, 11},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no stride
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyStride) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride length is 4
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyStrideLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//dilation length is no 5
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyDilationLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pads
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyPads) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pads length is not 6
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyPadsLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyPadsValue) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, -1, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// no input size
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyInputSize) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// length of input size is not 5
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDTestVerifyInputSizeLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10, 12});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// length of output padding is not 5
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDVerifyOutputPaddingLength) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
