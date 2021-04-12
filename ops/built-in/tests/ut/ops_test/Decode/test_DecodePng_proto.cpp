#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class decodepng : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "decodepng SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "decodepng TearDown" << std::endl;
  }
};

TEST_F(decodepng, decodepng_infershape_diff_test){
  ge::op::DecodePng op;
  op.UpdateInputDesc("contents", create_desc({1}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_UINT8);
  op.SetAttr("channels", 1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("image");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
  std::vector<int64_t> expected_output_shape = {-1,-1,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(decodepng, decodepng_infershape_channel_err_test){
  ge::op::DecodePng op;
  op.UpdateInputDesc("contents", create_desc({1}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_UINT8);
  op.SetAttr("channels", 6);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
