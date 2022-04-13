#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "stateful_random_ops.h"
using namespace ge;
using namespace op;

class RngReadAndSkipV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RngReadAndSkipV2Test test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RngReadAndSkipV2Test test TearDown" << std::endl;
  }
};

TEST_F(RngReadAndSkipV2Test, verify_infershape_00) {
  ge::op::RngReadAndSkipV2 op;
  op.UpdateInputDesc("algorithm", create_desc({1}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RngReadAndSkipV2Test, verify_infershape_01) {
  ge::op::RngReadAndSkipV2 op;
  op.UpdateInputDesc("algorithm", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("delta", create_desc({1}, ge::DT_INT64));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RngReadAndSkipV2Test, verify_infershape_02) {
  ge::op::RngReadAndSkipV2 op;
  op.UpdateInputDesc("algorithm", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("delta", create_desc({}, ge::DT_UINT64));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

