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