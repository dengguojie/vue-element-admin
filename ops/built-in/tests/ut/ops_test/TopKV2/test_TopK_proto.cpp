#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

// ----------------TopK-------------------
class TopKV2ProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopKV2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopKV2 Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKV2ProtoTest, topkv2_verify_test) {
  ge::op::TopKV2 op;
  op.UpdateInputDesc("x", create_desc({1, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("k", create_desc({1}, ge::DT_INT32));
  op.SetAttr("sorted", true);
  op.SetAttr("dim", -1);
  op.SetAttr("largest", true);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
