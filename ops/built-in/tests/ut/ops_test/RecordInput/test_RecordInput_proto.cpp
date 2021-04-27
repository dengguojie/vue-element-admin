#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
using namespace ge;
using namespace op;

class RecordInputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RecordInput test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RecordInput test TearDown" << std::endl;
  }
};

TEST_F(RecordInputTest, infer_shape_00) {
  ge::op::RecordInput op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}