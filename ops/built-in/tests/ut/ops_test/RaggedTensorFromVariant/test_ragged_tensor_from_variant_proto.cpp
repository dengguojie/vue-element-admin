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
 * @file test_ragged_tensor_from_variant_proto.cpp
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
class RaggedTensorFromVariant : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedTensorFromVariant SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedTensorFromVariant TearDown" << std::endl;
  }
};

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_shape_fail_1) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.SetAttr("input_ragged_rank", 2);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_shape_fail_2) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2, 3, 4}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", -1);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_shape_fail_3) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2, 3, 4}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", 2);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_attr_fail_1) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", -2);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_attr_fail_2) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", 1);
  op.SetAttr("output_ragged_rank", -1);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_attr_fail_3) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_attr_fail_4) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_shape_success_1) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(3);
  op.SetAttr("input_ragged_rank", 1);
  op.SetAttr("output_ragged_rank", 2);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
} 

TEST_F(RaggedTensorFromVariant, RaggedTensorFromVariant_infer_shape_success_2) {
  ge::op::RaggedTensorFromVariant op;
  op.UpdateInputDesc("encoded_ragged", create_desc({2, 3, 4}, ge::DT_VARIANT));
  op.create_dynamic_output_output_nested_splits(4);
  op.SetAttr("input_ragged_rank", -1);
  op.SetAttr("output_ragged_rank", 3);
  op.SetAttr("Tvalues", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}