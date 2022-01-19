#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArraySplit : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArraySplit Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArraySplit Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArraySplit, tensorArraySplit_infershape_input0_rank_fail){
  ge::op::TensorArraySplit op;
  op.UpdateInputDesc("handle", create_desc({4}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArraySplit, tensorArraySplit_infershape_input2_rank_fail){
  ge::op::TensorArraySplit op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("lengths", create_desc({}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArraySplit, tensorArraySplit_infershape_input3_rank_fail){
  ge::op::TensorArraySplit op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("lengths", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("flow_in", create_desc({4}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArraySplit, tensorArraySplit_infershape_success){
  ge::op::TensorArraySplit op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("lengths", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}