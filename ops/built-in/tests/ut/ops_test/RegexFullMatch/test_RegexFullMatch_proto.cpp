#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "string_ops.h"
using namespace ge;
using namespace op;

class RegexFullMatchTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RegexFullMatch test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RegexFullMatch test TearDown" << std::endl;
  }
};

TEST_F(RegexFullMatchTest, infer_shape_00) {
  ge::op::RegexFullMatch op;
  op.UpdateInputDesc("pattern", create_desc({}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegexFullMatchTest, infer_shape_01) {
  ge::op::RegexFullMatch op;
  op.UpdateInputDesc("pattern", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}