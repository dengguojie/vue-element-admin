#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "stateless_random_ops.h"

class StatelessTruncatedNormalV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessTruncatedNormalV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessTruncatedNormalV2 TearDown" << std::endl;
  }
};

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_test01) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  ge::TensorDesc y = op.GetOutputDescByName("y");

  EXPECT_EQ(y.GetShape().GetDimNum(), 3);
  EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_infer_failed_1) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({3}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_infer_failed_2) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_infer_failed_3) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_verify_failed_1) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT8));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_verify_failed_2) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT8));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_verify_failed_3) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT8));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_verify_failed_4) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT8));
  op.SetAttr("dtype", ge::DT_FLOAT);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2_verify_failed_5) {
  ge::op::StatelessTruncatedNormalV2 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({1}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT16);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
