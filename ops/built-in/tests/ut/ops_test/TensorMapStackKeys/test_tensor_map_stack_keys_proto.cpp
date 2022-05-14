/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_tensor_map_stack_keys_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/operator.h"
#include "map_ops.h"
#include "op_proto_test_common.h"
#include "op_proto_test_util.h"

class TENSOR_MAP_STACK_KEYS_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TENSOR_MAP_STACK_KEYS_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TENSOR_MAP_STACK_KEYS_UT TearDown" << std::endl;
  }
};

TEST_F(TENSOR_MAP_STACK_KEYS_UT, StackKeysInferShape_DT_INT64) {
  ge::op::TensorMapStackKeys op;
  op.SetAttr("key_dtype", ge::DT_INT64);
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr =
      std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("keys");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT64);
}

TEST_F(TENSOR_MAP_STACK_KEYS_UT, StackKeysInferShape_DT_INT32) {
  ge::op::TensorMapStackKeys op;
  op.SetAttr("key_dtype", ge::DT_INT32);
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr =
      std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("keys");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(TENSOR_MAP_STACK_KEYS_UT, StackKeysInferShape_DT_STRING) {
  ge::op::TensorMapStackKeys op;
  op.SetAttr("key_dtype", ge::DT_STRING);
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr =
      std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("keys");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_STRING);
}

TEST_F(TENSOR_MAP_STACK_KEYS_UT, StackKeysInferShape_ERROR) {
  ge::op::TensorMapStackKeys op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr =
      std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
