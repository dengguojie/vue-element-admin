#include <gtest/gtest.h>
#include <iostream>
#include "nn_calculation_ops.h"
#include "op_proto_test_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 10, 2, 2, 10}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{2, 10, 2, 2, 10},ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {10, 4, 1, 2, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{10, 4, 1, 2, 3},ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 10, 4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{1, 10, 4, 6, 8},ge::FORMAT_NCDHW));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {10, 1, 2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{10, 1, 2, 3, 4},ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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

// check stride length
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDStrideLen) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 10, 2, 2, 10}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{2, 10, 2, 2, 10},ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {10, 4, 1, 2, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{10, 4, 1, 2, 3},ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 10, 4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,{1, 10, 4, 6, 8},ge::FORMAT_NCDHW));
    op.SetAttr("input_size", {1, 10, 4, 6, 8});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// output_padding len
TEST_F(Conv3DTransposeDProtoTest, Conv3DTransposeDTestOutputpaddingLen) {
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {1, 4, 6, 8, 10});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0});
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10, 11}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10, 11},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 2, 2, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
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

// infer data slice --- empty query
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_query_empty){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- query list size larger than one
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_query_more_than_one){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{0,1}, {0,1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- cut N
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_cut_n){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{0, 1}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{0,1}, {}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// infer data slice --- cut D
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_cut_d){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 4}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {0, 2}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// infer data slice --- cut H
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_cut_h){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {0, 4}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {0, 2}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// infer data slice --- cut W
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_cut_w){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {0, 4}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {}, {0, 2}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// infer data slice --- cut Cout
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_cut_cout){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);
    std::vector<std::vector<int64_t>> expect_w_data_slice = {{0, 4}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}

// infer data slice --- stride len
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_stride_len){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- stride val
TEST_F(Conv3DTransposeDProtoTest, conv3dbpD_infer_data_slice_stride_val){
    ge::op::Conv3DTransposeD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
        {2, 4, 14, 14, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 4, 14, 14, 1024},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
        {2, 2, 2, 256, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{2, 2, 2, 256, 1024},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
        {2, 8, 28, 28, 512}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 8, 28, 28, 512},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 512});
    op.SetAttr("strides", {1, 2, -2, -2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}