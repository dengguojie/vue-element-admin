#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArrayGather : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayGather Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayGather Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayGather, tensorArrayGather_infershape_diff_test){
  ge::op::TensorArrayGather op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_input0_rank_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_input1_rank_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_input2_rank_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({4}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_context_null_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_attr_element_shape_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  inferCtxPtr->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.SetInferenceContext(inferCtxPtr);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_attr_dtype_fail){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  inferCtxPtr->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.SetInferenceContext(inferCtxPtr);
  op.SetAttr("element_shape", {4});
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayGather, tensorArrayGather_infershape_success){
  ge::op::TensorArrayGather op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  inferCtxPtr->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.SetInferenceContext(inferCtxPtr);
  op.SetAttr("element_shape", {4});
  op.SetAttr("dtype", ge::DT_INT64);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}