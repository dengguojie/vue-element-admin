#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"


class DenseImageWarpProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DenseImageWarp Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DenseImageWarp Proto Test TearDown" << std::endl;
  }
};


TEST_F(DenseImageWarpProtoTest, dense_image_warp_test) {
    ge::op::DenseImageWarp op;
    op.UpdateInputDesc("image", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 16, 64, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 2}, ge::FORMAT_NHWC));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DenseImageWarpProtoTest, dense_image_warp_test_format) {
    ge::op::DenseImageWarp op;
    op.UpdateInputDesc("image", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 2, 16, 64}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 2, 16, 64}, ge::FORMAT_NCHW));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DenseImageWarpProtoTest, dense_image_warp_test_flow) {
    ge::op::DenseImageWarp op;
    op.UpdateInputDesc("image", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DenseImageWarpProtoTest, dense_image_warp_test_size1) {
    ge::op::DenseImageWarp op;
    op.UpdateInputDesc("image", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 16, 32, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 32, 2}, ge::FORMAT_NHWC));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DenseImageWarpProtoTest, dense_image_warp_test_size2) {
    ge::op::DenseImageWarp op;
    op.UpdateInputDesc("image", create_desc_with_ori({4, 1, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 1, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 1, 64, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 1, 64, 2}, ge::FORMAT_NHWC));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}