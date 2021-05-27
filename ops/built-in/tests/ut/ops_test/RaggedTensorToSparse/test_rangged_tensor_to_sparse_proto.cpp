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
 * @file test_rangged_tensor_to_sparse_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "ragged_conversion_ops.h"
class RaggedTensorToSparseTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedTensorToSparse SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedTensorToSparse TearDown" << std::endl;
  }
};

TEST_F(RaggedTensorToSparseTest, InferShape_01) {
  ge::op::RaggedTensorToSparse op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.SetAttr("RAGGED_RANK", 0);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  auto tensor_desc = create_desc({4, 3},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_rt_nested_splits(1);
  op.UpdateDynamicInputDesc("rt_nested_splits", 0, tensor_desc);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.SetAttr("RAGGED_RANK", 1);
  tensor_desc = create_desc({4}, ge::DT_FLOAT16);
  op.create_dynamic_input_rt_nested_splits(1);
  op.UpdateDynamicInputDesc("rt_nested_splits", 0, tensor_desc);
  op.UpdateInputDesc("rt_dense_values", create_desc({2,3}, ge::DT_INT32));
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}