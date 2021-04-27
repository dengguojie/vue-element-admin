#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "stateful_random_ops.h"
using namespace ge;
using namespace op;

class RngSkipTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RngSkip test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RngSkip test TearDown" << std::endl;
  }
};

TEST_F(RngSkipTest, infer_shape_00) {
  ge::op::RngSkip op;
  op.UpdateInputDesc("algorithm", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RngSkipTest, infer_shape_01) {
  ge::op::RngSkip op;
  op.UpdateInputDesc("algorithm", create_desc({}, ge::DT_INT64));
  op.UpdateInputDesc("delta", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
