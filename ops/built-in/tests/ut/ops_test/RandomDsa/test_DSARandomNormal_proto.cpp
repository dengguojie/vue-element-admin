#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "randomdsa_ops.h"
using namespace ge;
using namespace op;

class DsaRandomNormalTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DsaRandomNormal test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DsaRandomNormal test TearDown" << std::endl;
  }
};

TEST_F(DsaRandomNormalTest, infer_shape_00) {
  ge::op::DSARandomNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomNormalTest, infer_shape_01) {
  ge::op::DSARandomNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({10, 10}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomNormalTest, infer_shape_02) {
  ge::op::DSARandomNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({10, 10}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomNormalTest, infer_shape_03) {
  ge::op::DSARandomNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
