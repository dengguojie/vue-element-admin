#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "ctc_ops.h"
#include "inference_context.h"

class cTCGreedyDecoder : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cTCGreedyDecoder Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cTCGreedyDecoder Proto Test TearDown" << std::endl;
  }
};

TEST_F(cTCGreedyDecoder, cTCGreedyDecoder_infershape_input0_rank_failed){
  ge::op::CTCGreedyDecoder op;
  op.UpdateInputDesc("inputs", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(cTCGreedyDecoder, cTCGreedyDecoder_infershape_input1_rank_failed){
  ge::op::CTCGreedyDecoder op;
  op.UpdateInputDesc("inputs", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("sequence_length", create_desc({}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}