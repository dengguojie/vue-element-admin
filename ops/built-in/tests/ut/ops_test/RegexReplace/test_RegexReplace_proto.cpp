#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "string_ops.h"
using namespace ge;
using namespace op;

class RegexReplaceTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RegexReplace test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RegexReplace test TearDown" << std::endl;
  }
};

TEST_F(RegexReplaceTest, infer_shape_00) {
  ge::op::RegexReplace op;
  op.UpdateInputDesc("pattern", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("rewrite", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegexReplaceTest, infer_shape_01) {
  ge::op::RegexReplace op;
  op.UpdateInputDesc("pattern", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegexReplaceTest, infer_shape_02) {
  ge::op::RegexReplace op;
  op.UpdateInputDesc("pattern", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("rewrite", create_desc({}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}