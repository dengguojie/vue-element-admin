#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "stateless_random_ops.h"

class StatelessRandomBinomial : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomBinomial SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomBinomial TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_test01) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({1, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("counts", create_desc({4, 1}, ge::DT_FLOAT));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  ge::TensorDesc y = op.GetOutputDescByName("y");

  EXPECT_EQ(y.GetShape().GetDimNum(), 3);
  EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_test02) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({52, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({1, 7, 35}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  ge::TensorDesc y = op.GetOutputDescByName("y");

  EXPECT_EQ(y.GetShape().GetDimNum(), 5);
  EXPECT_EQ(y.GetDataType(), ge::DT_INT32);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_infer_failed_1) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({2, 7, 1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_infer_failed_2) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({1, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("counts", create_desc({4, 1}, ge::DT_FLOAT));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_infer_failed_3) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({1, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("counts", create_desc({4, 1}, ge::DT_FLOAT));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_verify_failed_1) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({3, 7, 1}, ge::DT_INT64));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_verify_failed_2) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT8));
  op.UpdateInputDesc("counts", create_desc({3, 7, 1}, ge::DT_INT8));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_verify_failed_3) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT8));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({3, 7, 1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_verify_failed_4) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT8));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({3, 7, 1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT32);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomBinomial, StatelessRandomBinomial_verify_failed_5) {
  ge::op::StatelessRandomBinomial op;
  op.UpdateInputDesc("shape", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("probs", create_desc({3, 7, 1}, ge::DT_INT32));
  op.UpdateInputDesc("counts", create_desc({3, 7, 1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT8);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
