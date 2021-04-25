#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class LookupTableInsertTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LookupTableInsertTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LookupTableInsertTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(LookupTableInsertTest, LookupTableInsertTest_handle_error) {
  ge::op::LookupTableInsert op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
