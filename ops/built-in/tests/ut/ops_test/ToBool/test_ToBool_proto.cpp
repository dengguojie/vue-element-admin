#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "functional_ops.h"

// ----------------ToBool-------------------
class ToBoolTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ToBool Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ToBool Proto Test TearDown" << std::endl;
  }
};

TEST_F(ToBoolTest, ToBool_verify_test) {
  ge::op::ToBool op;
  op.UpdateOutputDesc("output", create_desc({}, ge::DT_BOOL));
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}


TEST_F(ToBoolTest, ToBool_verify_test_failed) {
  ge::op::ToBool op;
  op.UpdateOutputDesc("output", create_desc({}, ge::DT_INT32));
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
