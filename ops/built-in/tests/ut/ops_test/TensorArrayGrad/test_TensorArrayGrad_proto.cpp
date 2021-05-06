#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArrayGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayGrad, tensorArrayGrad_infershape_input0_rank_test){
  ge::op::TensorArrayGrad op;
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGrad, tensorArrayGrad_infershape_input1_rank_test){
  ge::op::TensorArrayGrad op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({2}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGrad, tensorArrayGrad_infershape_context_null_test){
  ge::op::TensorArrayGrad op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
