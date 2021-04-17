#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


class DeformableOffsetsProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DeformableOffsets Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DeformableOffsets Proto Test TearDown" << std::endl;
  }
};


TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_same) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 216, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 32, 192, 192}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 32, 192, 192}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_modulated) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", false);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_offsets) {
//     ge::op::DeformableOffsets op;
//     op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
//     op.UpdateInputDesc("offsets", create_desc_with_ori({4, 32, 32, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 32, 32, 216}, ge::FORMAT_NHWC));
//     op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
//     op.SetAttr("strides", {1, 1, 1, 1});
//     op.SetAttr("pads", {1, 1, 1, 1});
//     op.SetAttr("ksize", {3, 3});
//     op.SetAttr("dilations", {1, 1, 1, 1});
//     op.SetAttr("data_format", "NHWC");
//     op.SetAttr("deformable_groups", 8);
//     op.SetAttr("modulated", false);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);
// }

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_valid) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 216, 62, 62}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 62, 62}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 32, 186, 186}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 32, 186, 186}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", false);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_ksize) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3, 16, 16});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_ksize2) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_strides) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {-1, -1, -1, -1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_pads) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_strides2) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_dg) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 7);
    op.SetAttr("modulated", true);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_format) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 64, 216}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DeformableOffsetsProtoTest, deformable_offsets_test_offsets) {
    ge::op::DeformableOffsets op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("offsets", create_desc_with_ori({64, 64, 216}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{64, 64, 216}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 192, 192, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 192, 192, 32}, ge::FORMAT_NHWC));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NHWC");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}