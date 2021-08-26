/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "data_flow_ops.h"
#include "op_proto_test_common.h"

class QueueTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "QueueTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "QueueTest TearDown" << std::endl;
  }
};

TEST_F(QueueTest, fifo_queue_infershape_test) {
  ge::op::FIFOQueue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  std::vector<int64_t> shape{16, 16, 3};
  ge::Operator::OpListListInt elem_shapes{shape, shape};
  op.SetAttr("shapes", elem_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, fifo_queue_no_shape_infershape_test_fail) {
  ge::op::FIFOQueue op;
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(QueueTest, fifo_queue_no_shape_infershape_test_success) {
  ge::op::FIFOQueue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queue_size_infershape_test) {
  ge::op::QueueSize op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queue_is_closed_infershape_test) {
  ge::op::QueueIsClosed op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queue_enqueue_infershape_test) {
  ge::op::QueueEnqueue op;
  std::vector<std::string> marks = {std::string("QueueEnqueueTest1")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(QueueTest, fifo_queue_no_shape_infershape_test_twice_success) {
  ge::op::QueueEnqueue op;
  std::vector<std::string> marks = {std::string("QueueEnqueueTest1")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto ret1 = op.InferShapeAndType();
  EXPECT_EQ(ret1, ge::GRAPH_SUCCESS);
}

//QueueEnqueueMany
TEST_F(QueueTest, fifo_queue_enqueuemany_success) {
  ge::op::QueueEnqueueMany op;
  std::vector<std::string> marks = {std::string("fifo_queue_enqueuemany_success")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);

  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);

  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


//
TEST_F(QueueTest, fifo_queue_queueDequeue_shapeTypes_success) {
  ge::op::QueueDequeue op;
  std::vector<std::string> marks = {std::string("fifo_queue_enqueuemany_success")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//QueueEnqueueMany
TEST_F(QueueTest, fifo_queue_enqueuemany_marks_success) {
  ge::op::QueueEnqueueMany op;
  std::vector<std::string> marks = {std::string("fifo_queue_enqueuemany_marks_success")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);

  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, fifo_queue_enqueue_dequeue_success){
  ge::op::QueueEnqueue op;
  std::vector<std::string> marks = {std::string("QueueEnqueueTest2")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr enqueueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  enqueueinferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(enqueueinferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto ret1 = op.InferShapeAndType();
  EXPECT_EQ(ret1, ge::GRAPH_SUCCESS);

  ge::op::QueueDequeue op_dequeue;
  ge::InferenceContextPtr dequeueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  dequeueinferCtxPtr->SetMarks(marks);
  op_dequeue.SetInferenceContext(dequeueinferCtxPtr);  
  op_dequeue.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_dequeue.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op_dequeue.SetAttr("component_types", component_types);
  op_dequeue.create_dynamic_output_components(2);
  ret = op_dequeue.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);  
}

TEST_F(QueueTest, queue_dequeue_many_infershape_test1) {
  ge::op::QueueDequeueMany op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queue_dequeue_many_infershape_test2) {
  ge::op::QueueDequeueMany op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);

  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
//QueueDequeueMany
TEST_F(QueueTest, queue_dequeue_many_infershape_marks) {
  ge::op::QueueDequeueMany op;
  std::vector<std::string> marks = {std::string("queue_dequeue_many_infershape_marks")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queue_enqueue_dequeue_many_infershape_marks) {
  ge::op::QueueEnqueueMany op;
  std::vector<std::string> marks = {std::string("queue_enqueue_dequeue_many_infershape_marks")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr enqueueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  enqueueinferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(enqueueinferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::QueueDequeueMany op_dequeue;
  ge::InferenceContextPtr dequeueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  dequeueinferCtxPtr->SetMarks(marks);
  op_dequeue.SetInferenceContext(dequeueinferCtxPtr);  
  op_dequeue.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_dequeue.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op_dequeue.SetAttr("component_types", component_types);
  op_dequeue.create_dynamic_output_components(2);
  ret = op_dequeue.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);  
}


TEST_F(QueueTest, queue_enqueue_dequeue_many_infershape_marks_no_eq) {
  ge::op::QueueEnqueueMany op;
  std::vector<std::string> marks = {std::string("queue_enqueue_dequeue_many_infershape_marks")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr enqueueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  enqueueinferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(enqueueinferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.create_dynamic_input_components(2);
  op.UpdateDynamicInputDesc("components", 0, create_desc({2}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("components", 1, create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::QueueDequeueMany op_dequeue;
  ge::InferenceContextPtr dequeueinferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  dequeueinferCtxPtr->SetMarks(marks);
  op_dequeue.SetInferenceContext(dequeueinferCtxPtr);  
  op_dequeue.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_dequeue.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ge::DT_INT32};
  op_dequeue.SetAttr("component_types", component_types);
  op_dequeue.create_dynamic_output_components(2);
  ret = op_dequeue.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);  
}

TEST_F(QueueTest, queue_dequeue_many_infershape_test_fail) {
  ge::op::QueueDequeueMany op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(QueueTest, queue_dequeue_many_infershape_test_fail2) {
  ge::op::QueueDequeueMany op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);

  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  // op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(QueueTest, queue_dequeue_many_infershape_test_fail3) {
  ge::op::QueueDequeueMany op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  ge::ShapeAndType shape_and_type(ge::Shape({2}), ge::DT_INT32);
  std::vector<ge::ShapeAndType> handle_shapes_and_types{shape_and_type, shape_and_type};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(shapes_and_types));
  op.SetInferenceContext(inferCtxPtr);

  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("n", create_desc({}, ge::DT_INT32));
  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_n(concat_dim);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(QueueTest, padding_fifo_queue_infershape_test) {
  ge::op::PaddingFIFOQueue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  std::vector<int64_t> shape{16, 16, 3};
  ge::Operator::OpListListInt elem_shapes{shape, shape};
  op.SetAttr("shapes", elem_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queueDequeue_infershape_test) {
  ge::op::QueueDequeue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("component_types", component_types);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, queueDequeue_infershape_test_failed) {
  ge::op::QueueDequeue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  // std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  // op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(QueueTest, priority_queue_infershape_test) {
  ge::op::PriorityQueue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  std::vector<int64_t> shape{16, 16, 3};
  ge::Operator::OpListListInt elem_shapes{shape, shape};
  op.SetAttr("shapes", elem_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueTest, random_shuffle_queue_infershape_test) {
  ge::op::RandomShuffleQueue op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("component_types", component_types);
  std::vector<int64_t> shape{16, 16, 3};
  ge::Operator::OpListListInt elem_shapes{shape, shape};
  op.SetAttr("shapes", elem_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}