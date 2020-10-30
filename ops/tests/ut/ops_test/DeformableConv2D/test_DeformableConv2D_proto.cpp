#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


// ---------------Conv2D-------------------
class DeformableConv2DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DeformableConv2D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DeformableConv2D Proto Test TearDown" << std::endl;
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseTest) {
    // input：nhwc， filters：hwcn, offsets:nhwc, outputs:nhwc
    std::cout << "deformabelconv2d deformableconv2dBaseTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseTest1) {
    // input:nhwc, filters:nchw, offsets:nhwc, output:nhwc
    std::cout << "deformableconv2dBaseTest1" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{1, 16, 1, 1},ge::FORMAT_NCHW));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseTest2) {
    // input：nchw， filters：hwcn, offsets:nhwc, outputs:nhwc
    std::cout << "deformabelconv2d deformableconv2dBaseTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64},ge::FORMAT_NCHW));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseTest3) {
    // input:nhwc, filter:hwcn, offsets:nhwc, output:nchw
    std::cout << "deformableconv2dBaseTest3" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseInputTest) {
    std::cout << "deformableconv2dBaseInputTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16,16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16,16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16,16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16,16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseFilterTest) {
    std::cout<< "deformableconv2dBaseFilterTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1,1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{1,1, 1, 16, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input filter format should be nchw,nhwc,hwcn
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseFilterTest2) {
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_ND,{1, 1, 16, 1},ge::FORMAT_ND));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input x dtype is same as filter dtype
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseDtypeTest) {
    std::cout << "deformableconv2dBaseDtypeTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_INT8, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseStridesTest) {
    std::cout << "deformableconv2dBaseStridesTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseDilationsTest) {
    std::cout << "deformableconv2dBaseDilationsTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// pads should be positive
TEST_F(DeformableConv2DProtoTest, deformableconv2dBasePadTest1) {
    std::cout << "deformableconv2dBasePadTest1" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBasePadTest) {
    std::cout << "deformableconv2dBasePadTest" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseStridesTest1) {
    std::cout << "deformableconv2dBaseStridesTest1" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseDilationsTest1) {
    std::cout << "deformableconv2dBaseDilationsTest1" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseStridesTest2) {
    std::cout << "deformableconv2dBaseStridesTest2" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseDilationsTest2) {
    std::cout << "deformableconv2dBaseDilationsTest2" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
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
TEST_F(DeformableConv2DProtoTest, deformableconv2dBaseformatTest2) {
    std::cout << "deformableconv2dBaseformatTest2" << std::endl;
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 64, 64, 16},ge::FORMAT_ND));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 64, 64, 16},ge::FORMAT_ND));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input offset should be 4d
TEST_F(DeformableConv2DProtoTest, deformableconv2doffsetsTest) {
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// output format should be NCHW or NHWC
TEST_F(DeformableConv2DProtoTest, deformableconv2dOutputBaseTest) {
    ge::op::DeformableConv2D conv2d;
    conv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateInputDesc("filter", create_desc_with_ori({1, 1, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{1, 1, 16, 1},ge::FORMAT_HWCN));
    conv2d.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    conv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_ND,{4,64,64,1},ge::FORMAT_ND));
    conv2d.SetAttr("strides", {1, 1, 1, 1});
    conv2d.SetAttr("pads", {0, 0, 0, 0});
    conv2d.SetAttr("dilations", {1, 1, 1, 1});
    auto status = conv2d.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = conv2d.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}