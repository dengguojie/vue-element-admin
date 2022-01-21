#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "randomdsa_ops.h"
using namespace ge;
using namespace op;

class DsaRandomTruncatedNormalTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DsaRandomTruncatedNormal test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DsaRandomTruncatedNormal test TearDown" << std::endl;
  }
};

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_00) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_01) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({10, 10}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_02) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({10, 10}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_03) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
