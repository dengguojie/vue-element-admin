#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"


class DenseImageWarpGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DenseImageWarpGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DenseImageWarpGrad Proto Test TearDown" << std::endl;
  }
};


TEST_F(DenseImageWarpGradProtoTest, dense_image_warp_grad_test) {
    ge::op::DenseImageWarpGrad op;
    op.UpdateInputDesc("grad", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("image", create_desc_with_ori({4, 16, 64, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 3}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("flow", create_desc_with_ori({4, 16, 64, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 16, 64, 2}, ge::FORMAT_NHWC));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
