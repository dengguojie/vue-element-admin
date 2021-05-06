#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArrayGradWithShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayGradWithShape Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayGradWithShape Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_input0_rank_fail){
  ge::op::TensorArrayGradWithShape op;
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_input1_rank_fail){
  ge::op::TensorArrayGradWithShape op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_input3_rank_fail){
  ge::op::TensorArrayGradWithShape op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({4}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_context_null_fail){
  ge::op::TensorArrayGradWithShape op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}