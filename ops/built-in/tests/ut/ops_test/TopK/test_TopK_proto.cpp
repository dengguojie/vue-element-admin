#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

// ----------------AvgPool3D-------------------
class TopKProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopK Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopK Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKProtoTest, topk_verify_test) {
  ge::op::TopK op;
  op.UpdateInputDesc("x", create_desc({1, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("k", create_desc({1}, ge::DT_INT32));
  op.SetAttr("sorted", true);
  op.SetAttr("dim", -1);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
