#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
using namespace ge;
using namespace op;

class ResourceAccumulatorApplyGradientTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResourceAccumulatorApplyGradient test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResourceAccumulatorApplyGradient test TearDown" << std::endl;
  }
};

TEST_F(ResourceAccumulatorApplyGradientTest, infer_shape_00) {
  ge::op::ResourceAccumulatorApplyGradient op;
  op.UpdateInputDesc("local_step", create_desc({}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResourceAccumulatorApplyGradientTest, infer_shape_01) {
  ge::op::ResourceAccumulatorApplyGradient op;
  op.UpdateInputDesc("local_step", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}