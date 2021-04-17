#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


class DeformableOffsetsGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DeformableOffsetsGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DeformableOffsetsGrad Proto Test TearDown" << std::endl;
  }
};


TEST_F(DeformableOffsetsGradProtoTest, deformable_offsets_grad_test) {
    ge::op::DeformableOffsetsGrad op;
    op.UpdateInputDesc("grad", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 216, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("grad_x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("grad_offsets", create_desc_with_ori({4, 216, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 64, 64}, ge::FORMAT_NCHW));
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

TEST_F(DeformableOffsetsGradProtoTest, deformable_offsets_grad_test_shape) {
    ge::op::DeformableOffsetsGrad op;
    op.UpdateInputDesc("grad", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("offsets", create_desc_with_ori({4, 216, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("grad_x", create_desc_with_ori({4, 16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 16, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("grad_offsets", create_desc_with_ori({4, 216, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 216, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ksize", {3, 3});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("deformable_groups", 8);
    op.SetAttr("modulated", true);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
