#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

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
    deconv.SetAttr("data_format","NCHW");
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
    deconv.SetAttr("data_format","NCHW");
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// base dyanmic ut
TEST_F(DeconvProtoTest, deconvDynamicBaseTestFp16) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_shape_range({1, 16,  -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 1}, {16, 16}, {6, 26}, {6, 26}}));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_shape_range({1, 16,  -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 1}, {16, 16}, {6, 26}, {6, 26}}));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    auto status = deconv.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic nwc ut
TEST_F(DeconvProtoTest, deconvDynamicNWC) {
    ge::op::Deconvolution op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.SetAttr("strides", {1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    op.SetAttr("groups", 1);
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut outbackprop shape [-2]
TEST_F(DeconvProtoTest, deconvDynamicRank) {
    ge::op::Deconvolution op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
                       create_desc_shape_range({-2},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-2},
                                               ge::FORMAT_NCHW,
                                               {{}}));
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {1, -1}, {1, -1}}));
    op.SetAttr("strides", {1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
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
// cut Cin
TEST_F(DeconvProtoTest, deconvSplicDataTest0) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 2}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_w_data_slice = {{0, 26}, {}, {}, {}};
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}

// cut N
TEST_F(DeconvProtoTest, deconvSplicDataTest1) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    std::vector<std::vector<int64_t>> y_data_slice ={{1, 3}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{1, 3}, {}, {}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// cut H
TEST_F(DeconvProtoTest, deconvSplicDataTest2) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 5}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::int64_t> new_pads;
    deconv.GetAttr("pads", new_pads);
    std::vector<std::int64_t> new_pads_expect = {0, 2, 0, 0};
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 5}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(new_pads_expect, new_pads);
}

// cut w
TEST_F(DeconvProtoTest, deconvSplicDataTest3) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({4, 64, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 5, 5}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 13, 13}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 13, 13},ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {3, 3});
    deconv.SetAttr("pads", {1, 1, 1, 1});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {3, 10}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::int64_t> new_pads;
    deconv.GetAttr("pads", new_pads);
    std::vector<std::int64_t> new_pads_expect = {1, 1, 1, 0};
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {1, 3}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(new_pads_expect, new_pads);
}

// stride can not less than zero
TEST_F(DeconvProtoTest, deconvDataSliceTest2) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {-1, -1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {6, 20}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto ret = op_desc->InferDataSlice();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no data slice
TEST_F(DeconvProtoTest, deconvDataSliceTest3) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    auto ret = op_desc->InferDataSlice();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// shape dim check failed
TEST_F(DeconvProtoTest, deconvDataSliceTest4) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {6}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto ret = op_desc->InferDataSlice();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// cannot support cut in block_C
TEST_F(DeconvProtoTest, deconvDataSliceTest5) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {6, 20}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto ret = op_desc->InferDataSlice();
    EXPECT_EQ(ret, 50331645);
}

// no data slice
TEST_F(DeconvProtoTest, deconvDataSliceTest6) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");
    
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(deconv);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto ret = op_desc->InferDataSlice();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// y format should be NCHW
TEST_F(DeconvProtoTest, deconvOutputFormatTest) {
    ge::op::Deconvolution deconv;
    deconv.UpdateInputDesc("x", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 4, 4}, ge::FORMAT_NCHW));
    deconv.UpdateInputDesc("filter", create_desc_with_ori({16, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 16, 1, 1}, ge::FORMAT_NCHW));
    deconv.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_CHWN, {1, 16, 4, 4}, ge::FORMAT_CHWN));
    deconv.SetAttr("strides", {1, 1});
    deconv.SetAttr("pads", {0, 0, 0, 0});
    deconv.SetAttr("dilations", {1, 1, 1, 1});
    deconv.SetAttr("data_format","NCHW");

    auto ret = deconv.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}