#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------Conv2DCompress-------------------
class Conv2DCompressProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DCompress Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DCompress Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input x format shoudl be NCHW or NHWC
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest1) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{4, 64, 64, 16},ge::FORMAT_HWCN));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input filter format should be NCHW, NHWC or HWCN
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest2) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 16, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{1, 16, 1, 1, 1},ge::FORMAT_NC1HWC0));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// output y format should be NCHW or NHWC
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest3) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{4,64,64,1},ge::FORMAT_HWCN));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input x shape is empty
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest4) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// input x shape != 4
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest5) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// input w shape is empty
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest6) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// input w shape != 4
TEST_F(Conv2DCompressProtoTest, Conv2DCompressBaseTest7) {
    ge::op::Conv2DCompress op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter_compress", create_desc_with_ori({1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16},ge::FORMAT_HWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}