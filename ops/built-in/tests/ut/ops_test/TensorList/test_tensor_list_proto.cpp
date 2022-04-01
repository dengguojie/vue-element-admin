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
 * @file test_tensor_list_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/operator.h"
#include "op_proto_test_util.h"
#include "op_proto_test_common.h"
#include "list_ops.h"

class TENSOR_LIST_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TENSOR_LIST_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TENSOR_LIST_UT TearDown" << std::endl;
  }
};

TEST_F(TENSOR_LIST_UT, EmptyInferShape) {
  ge::op::EmptyTensorList op;
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("max_num_elements", create_desc({}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);

}

TEST_F(TENSOR_LIST_UT, PushBackInferShape) {
  ge::op::TensorListPushBack op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, PopBackInferShape) {
  ge::op::TensorListPopBack op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y1_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y1_desc.GetDataType(), ge::DT_VARIANT);

  auto y2_desc = op.GetOutputDesc("tensor");
  EXPECT_EQ(y2_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_LIST_UT, LengthInferShape) {
  ge::op::TensorListLength op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("length");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_LIST_UT, ElementShapeInferShape) {
  ge::op::TensorListElementShape op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.SetAttr("shape_type", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("element_shape");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_LIST_UT, ReserveInferShape) {
  ge::op::TensorListReserve op;
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("num_elements", create_desc({}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto reserve_desc = op.GetOutputDesc("handle");
  EXPECT_EQ(reserve_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, SetItemInferShape) {
  ge::op::TensorListSetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("item", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, SetItemInferShapeMarksFail) {
  std::vector<std::string> marks = {std::string("SetItemInferShapeMarksFail")};
  ge::op::TensorListSetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("item", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TENSOR_LIST_UT, SetItemInferShapeMarksEmpty) {
  std::vector<std::string> marks = {std::string("SetItemInferShapeMarksEmpty")};
  ge::op::TensorListSetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("item", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TENSOR_LIST_UT, GetItemInferShape) {
  ge::op::TensorListGetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto get_item_desc = op.GetOutputDesc("item");
  EXPECT_EQ(get_item_desc.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> expected_y_shape = {2, 2};
  EXPECT_EQ(get_item_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(TENSOR_LIST_UT, GetItemInferShapeMarksOk) {
  std::vector<std::string> marks = {std::string("GetItemInferShapeMarksOk")};

  ge::op::TensorListSetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("item", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);

  ge::op::TensorListGetItem op2;
  op2.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op2.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op2.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetMarks(marks);
  op2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
  auto get_item_desc = op2.GetOutputDesc("item");
  EXPECT_EQ(get_item_desc.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> expected_y_shape = {2, 2};
  EXPECT_EQ(get_item_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(TENSOR_LIST_UT, PushBackBatchInferShape) {
  ge::op::TensorListPushBackBatch op;
  op.UpdateInputDesc("input_handles", create_desc({2}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handles");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, PushBackBatchInferErrorShape) {
  ge::op::TensorListPushBackBatch op;
  op.UpdateInputDesc("input_handles", create_desc({2}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

}

TEST_F(TENSOR_LIST_UT, PushBackBatchInferShapeNoInputHandle) {
  ge::op::TensorListPushBackBatch op;
  op.UpdateInputDesc("input_handles", create_desc({2}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  key_value_vec.clear();
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TENSOR_LIST_UT, StackInferShape) {
  ge::op::TensorListStack op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("tensor");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}
TEST_F(TENSOR_LIST_UT, StackInferShapeMarksOk) {


std::vector<std::string> marks = {std::string("GetItemInferShapeMarksOk")};

  ge::op::TensorListSetItem op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("item", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);

  ge::op::TensorListStack op2;
  op2.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op2.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op2.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetMarks(marks);
  op2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
  auto y_desc2 = op2.GetOutputDesc("tensor");
  EXPECT_EQ(y_desc2.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_LIST_UT, StackInferShapeMarksFail) {
  std::vector<std::string> marks = {std::string("StackInferShapeMarksFail")};
  ge::op::TensorListStack op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("element_shape", create_desc({2, 2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TENSOR_LIST_UT, ConcatV2InferShape) {
  ge::op::TensorListConcatV2 op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("leading_dims", create_desc({}, ge::DT_INT64));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("tensor");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);

  auto length_desc = op.GetOutputDesc("lengths");
  EXPECT_EQ(length_desc.GetDataType(), ge::DT_INT64);
}

TEST_F(TENSOR_LIST_UT, SplitInferShape) {
  ge::op::TensorListSplit op;
  op.UpdateInputDesc("tensor", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("lengths", create_desc({1}, ge::DT_INT64));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, SplitInferErrorShape) {
  ge::op::TensorListSplit op;
  op.UpdateInputDesc("tensor", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("lengths", create_desc({1}, ge::DT_INT64));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

}

TEST_F(TENSOR_LIST_UT, FromTensorInferShape) {
  ge::op::TensorListFromTensor op;
  op.UpdateInputDesc("tensor", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, ResizeInferShape) {
  ge::op::TensorListResize op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, GatherInferShape) {
  ge::op::TensorListGather op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("values");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_LIST_UT, ScatterV2InferShape) {
  ge::op::TensorListScatterV2 op;
  op.UpdateInputDesc("tensor", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("element_shape", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("num_elements", create_desc({}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, ScatterIntoExistingListInferShape) {
  ge::op::TensorListScatterIntoExistingList op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({2,2}, ge::DT_INT32));
  op.UpdateInputDesc("indices", create_desc({1}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}

TEST_F(TENSOR_LIST_UT, ScatterIntoExistingListInferErrorShape) {
  ge::op::TensorListScatterIntoExistingList op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("tensor", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("indices", create_desc({1}, ge::DT_INT32));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

}

TEST_F(TENSOR_LIST_UT, ConcatListsInferShape) {
  ge::op::TensorListConcatLists op;
  op.UpdateInputDesc("input_a", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("input_b", create_desc({}, ge::DT_VARIANT));
  op.SetAttr("element_dtype", ge::DT_INT32);
  ge::InferenceContextPtr inferCtxPtr = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2, 2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("output");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}