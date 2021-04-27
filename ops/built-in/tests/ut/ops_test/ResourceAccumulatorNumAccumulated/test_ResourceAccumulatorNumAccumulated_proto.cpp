#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
using namespace ge;
using namespace op;

class ResourceAccumulatorNumAccumulatedTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResourceAccumulatorNumAccumulated test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResourceAccumulatorNumAccumulated test TearDown" << std::endl;
  }
};

TEST_F(ResourceAccumulatorNumAccumulatedTest, infer_shape_00) {
  ge::op::ResourceAccumulatorNumAccumulated op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}