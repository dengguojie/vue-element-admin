#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
using namespace ge;
using namespace op;

class ResourceConditionalAccumulatorTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResourceConditionalAccumulator test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResourceConditionalAccumulator test TearDown" << std::endl;
  }
};

TEST_F(ResourceConditionalAccumulatorTest, infer_shape_00) {
  ge::op::ResourceConditionalAccumulator op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResourceConditionalAccumulatorTest, infer_shape_01) {
  ge::op::ResourceConditionalAccumulator op;
  op.set_attr_dtype(ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
