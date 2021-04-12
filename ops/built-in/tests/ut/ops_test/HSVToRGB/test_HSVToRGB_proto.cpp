#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class hsvtorgb : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HSVToRGB SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HSVToRGB TearDown" << std::endl;
  }
};

TEST_F(hsvtorgb, hsvtorgb_infershape_input_rank_err_test){
  ge::op::HSVToRGB op;
  op.UpdateInputDesc("images", create_desc({}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(hsvtorgb, hsvtorgb_infershape_input_dim_err_test){
  ge::op::HSVToRGB op;
  op.UpdateInputDesc("images", create_desc({5}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}