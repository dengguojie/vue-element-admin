#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "bitwise_ops.h"
using namespace ge;
using namespace op;

class RightShiftTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RightShift test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RightShift test TearDown" << std::endl;
  }
};

TEST_F(RightShiftTest, infer_shape_00) {
  ge::op::RightShift op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
