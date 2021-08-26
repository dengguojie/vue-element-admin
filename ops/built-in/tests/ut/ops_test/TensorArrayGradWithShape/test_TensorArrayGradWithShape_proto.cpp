#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"
#include "op_proto_test_common.h"

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

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_context_success){
  ge::op::TensorArrayGradWithShape op;
  std::vector<std::string> marks = {std::string("tensorArrayGradWithShape_infershape_context_success")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({2}, ge::DT_FLOAT));
  op.UpdateInputDesc("shape_to_prepend", create_desc({2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayGradWithShape, tensorArrayGradWithShape_infershape_input_success){
  ge::op::TensorArrayGradWithShape op;
  std::vector<std::string> marks = {std::string("tensorArrayGradWithShape_infershape_context_success")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);

  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({2}, ge::DT_FLOAT));
  op.UpdateInputDesc("shape_to_prepend", create_desc({2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}