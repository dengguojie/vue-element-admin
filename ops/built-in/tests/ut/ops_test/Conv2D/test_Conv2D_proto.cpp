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

// ---------------Conv2D-------------------
class Conv2DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2D Proto Test TearDown" << std::endl;
  }
};

// REG_OP(Conv2D)
//     .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
//     .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
//     .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
//     .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
//     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
//     .REQUIRED_ATTR(strides, ListInt)
//     .REQUIRED_ATTR(pads, ListInt)
//     .ATTR(dilations, ListInt, {1, 1, 1, 1})
//     .ATTR(groups, Int, 1)
//     .ATTR(data_format, String, "NHWC")
//     .ATTR(offset_x, Int, 0)
//     .OP_END_FACTORY_REG(Conv2D)

// base ut
TEST_F(Conv2DProtoTest, conv2dBaseTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut
TEST_F(Conv2DProtoTest, conv2dBaseTest1) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{1, 16, 1, 1},ge::FORMAT_NCHW));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut
TEST_F(Conv2DProtoTest, conv2dBaseTest2) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_INT8, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_INT8, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_INT32, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut
TEST_F(Conv2DProtoTest, conv2dBaseTest3) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,1,64,64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4,1,64,64},ge::FORMAT_NCHW));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input x shape not 4D(case 5D)
TEST_F(Conv2DProtoTest, conv2dBaseInputTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16,16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16,16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,32,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,32,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 2, 2});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input filter shape shoule be 4D(case 5D)
TEST_F(Conv2DProtoTest, conv2dBaseFilterTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1,1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{1,1, 1, 16, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// in_channels should is divisible by kernel_channels when groups = 1
TEST_F(Conv2DProtoTest, conv2dGroupsInputTest1) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 4, 4},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,4}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,4},ge::FORMAT_NHWC));
    conv2d.SetAttr("groups", 1);
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// in_channels should is not divisible by kernel_channels when groups = 1
TEST_F(Conv2DProtoTest, conv2dGroupsInputTest2) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 10, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 10, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("groups", 1);
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input x dtype is same as filter dtype
TEST_F(Conv2DProtoTest, conv2dBaseDtypeTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_INT8, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// get strides list failed
TEST_F(Conv2DProtoTest, conv2dBaseStridesTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    // conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// get dilations list failed
TEST_F(Conv2DProtoTest, conv2dBaseDilationsTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// pads should be positive
TEST_F(Conv2DProtoTest, conv2dBasePadTest1) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {-1, -1, -1, -1});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pads list should be 4D
TEST_F(Conv2DProtoTest, conv2dBasePadTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides list should be 4D
TEST_F(Conv2DProtoTest, conv2dBaseStridesTest1) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations list should be 4D
TEST_F(Conv2DProtoTest, conv2dBaseDilationsTest1) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides should be positive
TEST_F(Conv2DProtoTest, conv2dBaseStridesTest2) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {-1, -1, -1, -1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations should be positive
TEST_F(Conv2DProtoTest, conv2dBaseDilationsTest2) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {-1, -1, -1, -1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input x format should be NCHW or NHWC
TEST_F(Conv2DProtoTest, conv2dBaseformatTest2) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 64, 64, 16},ge::FORMAT_ND));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// image size after padding should be greater than or equal to filter size.
TEST_F(Conv2DProtoTest, conv2dBasefmgreaterthanfilterTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Conv2DProtoTest, conv2dSplicDataTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({3, 3, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{3, 3, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {1, 1, 1, 1});
    conv2d.SetAttr("dilations", {2, 2, 2, 2});
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0,62}, {3,10}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(conv2d);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0,63}, {2,13}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    conv2d.GetAttr("pads",pads);
    std::vector<int> expect_pads = {1, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}

// base dynamic ut with range(6, -1)
TEST_F(Conv2DProtoTest, conv2dDynamicBaseTest) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_shape_range({4, -1, -1, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, -1, -1, 64},
                                                        ge::FORMAT_NHWC, {{4, 4}, {6, -1}, {6, -1}, {64, 64}}));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 64, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 64, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_shape_range({4, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, -1, -1, 1},
                                                        ge::FORMAT_NHWC, {{4, 4}, {6, -1}, {6, -1}, {1, 1}}));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base dynamic ut with range(6, -1) for padding mode "same"
TEST_F(Conv2DProtoTest, conv2dDynamicBaseTestPaddingMode) {
    ge::op::Conv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_shape_range({4, -1, -1, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, -1, -1, 64},
                                                        ge::FORMAT_NHWC, {{4, 4}, {6, -1}, {6, -1}, {64, 64}}));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 64, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 64, 1},ge::FORMAT_HWCN));
    conv2d.UpdateOutputDesc("y", create_desc_shape_range({4, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, -1, -1, 1},
                                                        ge::FORMAT_NHWC, {{4, 4}, {6, -1}, {6, -1}, {1, 1}}));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("padding", "SAME");
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}