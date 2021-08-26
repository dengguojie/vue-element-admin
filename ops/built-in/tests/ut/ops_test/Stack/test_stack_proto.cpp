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
 * @file test_stack_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "op_proto_test_common.h"

class STACK_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "STACK_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "STACK_UT TearDown" << std::endl;
  }
};


class Stack : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "STACK_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "STACK_UT TearDown" << std::endl;
  }
};

class StackPush : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "STACK_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "STACK_UT TearDown" << std::endl;
  }
};


TEST_F(STACK_UT, stack_infer_shape) {
  ge::op::StackPop op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(Stack, stack_infer_shape) {
  ge::op::Stack op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(STACK_UT, stack_pop_infer_shape) {
  ge::op::StackPop op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(STACK_UT, stack_pop_infer_shape_failed) {
  ge::op::StackPop op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  // op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(STACK_UT, stack_pop_infer_shape_failed2) {
  ge::op::StackPop op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  inferCtxPtr->SetMarks({std::string("testName")});
  std::vector<std::vector<ge::ShapeAndType>> key_value_vec;
  std::vector<ge::ShapeAndType> key_value;
  ge::DataType dataType = ge::DT_INT32;
  ge::Shape shape({2,2});
  ge::ShapeAndType key(shape, dataType);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  inferCtxPtr->SetInputHandleShapesAndTypes(std::move(key_value_vec));

  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(STACK_UT, stack_push_infer_shape) {
  ge::op::StackPush op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(STACK_UT, stack_push_infer_shape_marks) {
  ge::op::StackPush op;
  std::vector<std::string> marks = {std::string("stack_push_infer_shape_marks")};
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  //ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  inferCtxPtr->SetMarks(marks);
  op.SetInferenceContext(inferCtxPtr);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));

  std::vector<ge::DataType> component_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("elem_type", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(STACK_UT, stack_close_infer_shape) {
  ge::op::StackClose op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_RESOURCE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}