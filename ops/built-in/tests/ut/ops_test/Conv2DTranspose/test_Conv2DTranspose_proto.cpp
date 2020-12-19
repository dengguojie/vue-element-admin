#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"

// ---------------Conv2DTranspose-------------------
class Conv2DTransposeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DTranspose Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DTranspose Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyBaseTest) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("bias", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

//check fm type diff filter type
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyfilterTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_INT32));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//check filter out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyfilterTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check x size
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyInputTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no stride
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyStrideTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    //   op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyStrideTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no dilations
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyDilationsTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    //   op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyDilationsTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pad
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    //   op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest3) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no output_padding
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyOutputpaddingTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    //   op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// output_padding out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyOutputpaddingTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no input size
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeInferInputSizesTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    //   op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input size not 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyInputSizesTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


