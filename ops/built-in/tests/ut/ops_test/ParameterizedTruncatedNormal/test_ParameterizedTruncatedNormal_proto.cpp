#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"
using namespace ge;
using namespace op;

class ParameterizedTruncatedNormalTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ParameterizedTruncatedNormal test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ParameterizedTruncatedNormal test TearDown" << std::endl;
  }
};

TEST_F(ParameterizedTruncatedNormalTest, infer_shape_00) {
  ge::op::ParameterizedTruncatedNormal op;
  op.UpdateInputDesc("shape", create_desc({10, 10}, ge::DT_INT32));
  op.UpdateInputDesc("means", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("stdevs", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_DOUBLE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ParameterizedTruncatedNormalTest, infer_shape_01) {
  ge::op::ParameterizedTruncatedNormal op;
  op.UpdateInputDesc("shape", create_desc({10, 10}, ge::DT_INT32));
  op.UpdateInputDesc("means", create_desc({10, 10}, ge::DT_DOUBLE));
  op.UpdateInputDesc("stdevs", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_DOUBLE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ParameterizedTruncatedNormalTest, infer_shape_02) {
  ge::op::ParameterizedTruncatedNormal op;
  op.UpdateInputDesc("shape", create_desc({10, 10}, ge::DT_INT32));
  op.UpdateInputDesc("means", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("stdevs", create_desc({10, 10}, ge::DT_DOUBLE));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_DOUBLE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ParameterizedTruncatedNormalTest, infer_shape_03) {
  ge::op::ParameterizedTruncatedNormal op;
  op.UpdateInputDesc("shape", create_desc({10, 10}, ge::DT_INT32));
  op.UpdateInputDesc("means", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("stdevs", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("min", create_desc({10, 10}, ge::DT_DOUBLE));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_DOUBLE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ParameterizedTruncatedNormalTest, infer_shape_04) {
  ge::op::ParameterizedTruncatedNormal op;
  op.UpdateInputDesc("shape", create_desc({10, 10}, ge::DT_INT32));
  op.UpdateInputDesc("means", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("stdevs", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_DOUBLE));
  op.UpdateInputDesc("max", create_desc({10, 10}, ge::DT_DOUBLE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}