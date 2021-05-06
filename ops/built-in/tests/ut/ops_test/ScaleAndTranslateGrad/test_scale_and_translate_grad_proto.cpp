#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "image_ops.h"

class scale_and_translate_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scale_and_translate_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scale_and_translate_grad TearDown" << std::endl;
  }
};

TEST_F(scale_and_translate_grad, scale_and_translate_grad_infershape_test_1) {
  ge::op::ScaleAndTranslateGrad op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("original_image", create_desc_with_ori(
      {2, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 4, 2}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(scale_and_translate_grad, scale_and_translate_grad_infershape_test_2) {
  ge::op::ScaleAndTranslateGrad op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("original_image", create_desc_with_ori(
      {2, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 4, 2}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(scale_and_translate_grad, scale_and_translate_grad_infershape_test_3) {
  ge::op::ScaleAndTranslateGrad op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("original_image", create_desc_with_ori(
      {2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}