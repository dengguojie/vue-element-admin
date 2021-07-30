#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "logging_ops.h"

class PrintV3 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PrintV3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PrintV3 TearDown" << std::endl;
  }
};

TEST_F(PrintV3, PrintV3_infer_shape_0) {
    ge::op::PrintV3 op;
    op.UpdateInputDesc("x", create_desc({1}, ge::DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
