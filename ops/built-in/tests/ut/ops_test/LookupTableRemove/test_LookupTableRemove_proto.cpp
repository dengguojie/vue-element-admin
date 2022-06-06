#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class LookupTableRemove : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LookupTableRemove Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LookupTableRemove Proto Test TearDown" << std::endl;
  }
};

TEST_F(LookupTableRemove, LookupTableRemove_infer_failed_1) {
  ge::op::LookupTableRemove op;
  op.UpdateInputDesc("table_handle", create_desc({1}, ge::DT_RESOURCE));
  op.UpdateInputDesc("keys", create_desc({2}, ge::DT_BOOL));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableRemove, LookupTableRemove_success_1) {
  ge::op::LookupTableRemove op;
  op.UpdateInputDesc("table_handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("keys", create_desc({2}, ge::DT_BOOL));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
