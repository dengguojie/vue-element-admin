#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
using namespace ge;
using namespace op;

class ReverseSequenceTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReverseSequence test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReverseSequence test TearDown" << std::endl;
  }
};

TEST_F(ReverseSequenceTest, infer_shape_00) {
  ge::op::ReverseSequence op;
  op.UpdateInputDesc("seq_lengths", create_desc({}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ReverseSequenceTest, infer_shape_01) {
  ge::op::ReverseSequence op;
  op.UpdateInputDesc("seq_lengths", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x", create_desc({-1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ReverseSequenceTest, infer_shape_02) {
  ge::op::ReverseSequence op;
  op.UpdateInputDesc("seq_lengths", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x", create_desc({1}, ge::DT_INT64));
  op.set_attr_seq_dim(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ReverseSequenceTest, infer_shape_03) {
  ge::op::ReverseSequence op;
  op.UpdateInputDesc("seq_lengths", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x", create_desc({1}, ge::DT_INT64));
  op.set_attr_seq_dim(0);
  op.set_attr_batch_dim(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ReverseSequenceTest, infer_shape_04) {
  ge::op::ReverseSequence op;
  op.UpdateInputDesc("seq_lengths", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x", create_desc({1}, ge::DT_INT64));
  op.set_attr_seq_dim(0);
  op.set_attr_batch_dim(0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
