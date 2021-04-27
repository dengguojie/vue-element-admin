#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
using namespace ge;
using namespace op;

class ResourceAccumulatorSetGlobalStepTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResourceAccumulatorSetGlobalStep test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResourceAccumulatorSetGlobalStep test TearDown" << std::endl;
  }
};

TEST_F(ResourceAccumulatorSetGlobalStepTest, infer_shape_00) {
  ge::op::ResourceAccumulatorSetGlobalStep op;
  op.UpdateInputDesc("new_global_step", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResourceAccumulatorSetGlobalStepTest, infer_shape_01) {
  ge::op::ResourceAccumulatorSetGlobalStep op;
  op.UpdateInputDesc("new_global_step", create_desc({}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
