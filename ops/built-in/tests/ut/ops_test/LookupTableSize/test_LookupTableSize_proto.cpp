#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class LookupTableSizeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LookupTableSizeTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LookupTableSizeTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(LookupTableSizeTest, LookupTableSizeTest_success) {
  ge::op::LookupTableSize op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
