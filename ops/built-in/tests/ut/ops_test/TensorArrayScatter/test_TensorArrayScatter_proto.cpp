#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"
#include "../util/common_shape_fns.h"
#include "op_proto_test_common.h"

class tensorArrayScatter : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayScatter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayScatter Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_diff_test){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_input0_rank_failed){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_input3_rank_failed){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({3}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_input1_rank_failed){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_input2_rank_failed){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_context_null_failed){
  ge::op::TensorArrayScatter op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_input_handle_no_empty){
  ge::op::TensorArrayScatter op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());

  std::vector<std::vector<ge::ShapeAndType> > key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value.emplace_back(key);
  key_value_vec.push_back(key_value);
  key_value.clear();
  key_value.emplace_back(key);
  key_value.emplace_back(key);
  key_value_vec.push_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayScatter, tensorArrayScatter_input_handle_empty){
  ge::op::TensorArrayScatter op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}});
  op.UpdateInputDesc("value", value_desc);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  std::vector<std::string> marks = {std::string("tensorArray003")};
  inferCtxPtr->SetMarks(marks);
  ge::AicpuResourceContext *aicpu_resource_context = new ge::AicpuResourceContext();
  ge::Shape shape;
  std::vector<std::pair<int64_t, int64_t>> value_shape_range;
  ge::ShapeAndRange feed_shape_and_range{shape, value_shape_range};
  aicpu_resource_context->shape_and_range_.push_back(feed_shape_and_range);
  inferCtxPtr->SetResourceContext(marks[0].c_str(), aicpu_resource_context);
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}