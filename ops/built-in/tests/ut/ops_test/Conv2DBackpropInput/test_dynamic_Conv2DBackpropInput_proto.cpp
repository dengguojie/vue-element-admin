#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"

// ---------------Conv2DTransposeD-------------------
class Conv2DBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test TearDown" << std::endl;
  }
};


// dynamic opti ut with pads
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputOptiWithPads) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {6, 26}, {6, 26}}));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// dynamic opti ut with stride>1
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputOptiWithStride) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 3, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {1, 18}, {1, 18}}));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// dynamic general ut with stride>1
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputGeneWithStride) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 3, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {1, 18}, {1, 18}}));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// dynamic general ut with out_backprop format is NHWC
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputGeneWithxFormat) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 3, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, -1, -1, 32},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NHWC,
                                               {1, -1, -1, 32},
                                               ge::FORMAT_NHWC,
                                               {{1, 1}, {1, 18}, {1, 18}, {32, 32}}));
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("pads", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// dynamic general ut with dilations<0
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputGeneWithDilation) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 3, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {1, 18}, {1, 18}}));
    op.SetAttr("dilations", {-1, -1, -1, -1});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check fm type diff filter type
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyFilterTest1) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_INT32));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check filter out of size 4
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyFilterTest2) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check fm type diff out_backprop type
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyOutBackpropTest1) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_INT32));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check out_backprop size
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyOutBackpropTest2) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1, 1}, ge::DT_INT8));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyPadsTest) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));
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
}
