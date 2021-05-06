#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "image_ops.h"

class scale_and_translate : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scale_and_translate SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scale_and_translate TearDown" << std::endl;
  }
};

TEST_F(scale_and_translate, scale_and_translate_infershape_test_1) {
  ge::op::ScaleAndTranslate op;
  auto tensor_desc = create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("images", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}