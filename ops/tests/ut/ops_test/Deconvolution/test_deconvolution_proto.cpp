#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


// ---------------Deconv test proto-------------------
class DeconvProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Deconv Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Deconv Proto Test TearDown" << std::endl;
  }
};


// REG_OP(Deconvolution)
//    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
//    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
//    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
//    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
//    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
//    .REQUIRED_ATTR(strides, ListInt)
//    .REQUIRED_ATTR(pads, ListInt)
//    .ATTR(dilations, ListInt, {1, 1, 1, 1})
//    .ATTR(groups, Int, 1)
//    .ATTR(data_format, String, "NCHW")
//    .ATTR(offset_x, Int, 0)
//    .OP_END_FACTORY_REG(Deconvolution)


// base ut
TEST_F(DeconvProtoTest, deconvBaseTestFp16) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base ut
TEST_F(DeconvProtoTest, deconvBaseTestInt8) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_INT8, ge::FORMAT_NCHW,{4, 64, 64, 16}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_INT8, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_INT32, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input x shape not 4D(case 5D)
TEST_F(DeconvProtoTest, deconvBaseInputTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4, 16}, ge::DT_INT8, ge::FORMAT_NCHW,{4, 64, 64, 16, 16}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_INT8, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_INT32, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input filter shape shoule be 4D(case 5D)
TEST_F(DeconvProtoTest, deconvBaseFilterTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 64, 16}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1, 8}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1, 8}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input x dtype is same as filter dtype
TEST_F(DeconvProtoTest, deconvBaseDtypeTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_INT8, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// input ic == filter kn
TEST_F(DeconvProtoTest, deconvBaseChannelTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 32, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 32, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// get strides list failed
TEST_F(DeconvProtoTest, deconvBaseStrideTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// get dilations list failed
TEST_F(DeconvProtoTest, deconvBaseDilationTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// pads should be positive
TEST_F(DeconvProtoTest, deconvBasePadTest1) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {-1, -2, -1, -2});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pads list should be 4D
TEST_F(DeconvProtoTest, deconvBasePadTest2) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides list should be 2D
TEST_F(DeconvProtoTest, deconvBaseStideTest1) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides list should be positive
TEST_F(DeconvProtoTest, deconvBaseStideTest2) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {-1, -1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations list should be 4D
TEST_F(DeconvProtoTest, deconvBaseDilationTest1) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input x format should be NCHW
TEST_F(DeconvProtoTest, deconvBaseFormatTest1) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 16, 4, 4}, ge::FORMAT_NHWC));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NHWC));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// filter format should be NCHW
TEST_F(DeconvProtoTest, deconvBaseFormatTest2) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {16, 16, 1, 1}, ge::FORMAT_NHWC));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
