#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

// ----------------TopKD-------------------
class TopKDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopKD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopKD Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKDProtoTest, topk_verify_test) {
  ge::op::TopKD op;
  op.UpdateInputDesc("x", create_desc({1, 16}, ge::DT_FLOAT16));
  op.SetAttr("k", 1);
  op.SetAttr("sorted", true);
  op.SetAttr("dim", -1);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
