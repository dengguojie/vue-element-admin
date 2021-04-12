#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class encodepng : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "encodepng SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "encodepng TearDown" << std::endl;
  }
};

TEST_F(encodepng, encodepng_infershape_input_rank_err_test){
  ge::op::EncodePng op;
  op.UpdateInputDesc("image", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
