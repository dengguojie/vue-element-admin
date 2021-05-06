#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "ctc_ops.h"
#include "inference_context.h"

class cTCLoss : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cTCLoss Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cTCLoss Proto Test TearDown" << std::endl;
  }
};

TEST_F(cTCLoss, cTCLoss_infershape_input0_rank_failed){
  ge::op::CTCLoss op;
  op.UpdateInputDesc("inputs", create_desc({2}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cTCLoss, cTCLoss_infershape_input1_rank_failed){
  ge::op::CTCLoss op;
  op.UpdateInputDesc("inputs", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("labels_indices", create_desc({}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cTCLoss, cTCLoss_infershape_input2_rank_failed){
  ge::op::CTCLoss op;
  op.UpdateInputDesc("inputs", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("labels_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("labels_values", create_desc({}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cTCLoss, cTCLoss_infershape_input3_rank_failed){
  ge::op::CTCLoss op;
  op.UpdateInputDesc("inputs", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("labels_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("labels_values", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("sequence_length", create_desc({}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}