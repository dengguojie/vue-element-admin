#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "image_ops.h"
using namespace ge;
using namespace op;

class RGBToHSVTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RGBToHSV test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RGBToHSV test TearDown" << std::endl;
  }
};

TEST_F(RGBToHSVTest, infer_shape_00) {
  ge::op::RGBToHSV op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
