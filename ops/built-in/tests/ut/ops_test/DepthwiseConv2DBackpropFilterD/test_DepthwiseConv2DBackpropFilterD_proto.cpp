#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"


// ---------------DepthwiseConv2DBackpropFilterD-------------------
class DepthwiseConv2DBackpropFilterDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2DBackpropFilterD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2DBackpropFilterD Proto Test TearDown" << std::endl;
  }
};


// VALID NHWC
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDBase1) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 7, 7, 512},
        ge::DT_FLOAT16, ge::FORMAT_NHWC, {128, 7, 7, 512}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("input", create_desc_with_ori({128, 14, 14, 256},
        ge::DT_FLOAT16, ge::FORMAT_NHWC, {128, 14, 14, 256}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 1, 256, 512},
        ge::DT_FLOAT16, ge::FORMAT_HWCN, {1, 1, 256, 512}, ge::FORMAT_HWCN));

    op.SetAttr("filter_size", {1, 1, 256, 512});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NHWC");
    std::string padding = "VALID";
    op.SetAttr("padding", padding);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// SAME NCHW
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDBase2) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_HWCN, {1,1,256,512}, ge::FORMAT_HWCN));
    op.SetAttr("filter_size", {512, 1, 1, 256});
    op.SetAttr("strides", {2, 2, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NHWC");
    op.SetAttr("groups", 1);
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
// Error data_format
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDErrorDataFormat) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_HWCN, {1,1,256,512}, ge::FORMAT_HWCN));
    op.SetAttr("filter_size", {512, 1, 1, 256});
    op.SetAttr("strides", {2, 2, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NHW");
    op.SetAttr("groups", 1);
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check fm type diff x type
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyXDtypeTest) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_INT8, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check filter_size out of size 4
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyFilterSizeDimTest) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no stride
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyStrideTest1) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyStrideTest2) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyDilationsDimTest) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pad
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyPadsTest1) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyPadsTest2) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyPadsTest3) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// dataformat 
TEST_F(DepthwiseConv2DBackpropFilterDProtoTest, DepthwiseConv2DBackpropFilterDVerifyDataFormatTest) {
    ge::op::DepthwiseConv2DBackpropFilterD op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","HWCN");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);
    op.SetAttr("filter_size", {512, 256, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
