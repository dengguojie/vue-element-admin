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
 * @file test_concat_proto.cpp
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
#include "split_combination_ops.h"

class ConcatOffset : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatOffset SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatOffset TearDown" << std::endl;
  }
};


TEST_F(ConcatOffset, concat_offset_infer_1) {
  ge::op::ConcatOffset op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{4, 8}};
  auto tensor_desc = create_desc({4},
                                 ge::DT_INT32);
  auto tensor_desc_with_range = create_desc_shape_range({-1},
                                                        ge::DT_INT32,
                                                        ge::FORMAT_ND,
                                                        {-1},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_with_range);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  op.create_dynamic_output_y(3);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::cout << "ConcatOffset check" << std::endl;
  auto output_desc = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{4, 4}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatOffset, concat_offset_infer_2) {
  ge::op::ConcatOffset op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{4, 8}};
  auto tensor_desc = create_desc({4},
                                 ge::DT_INT32);
  auto tensor_desc_with_range = create_desc_shape_range({-1},
                                                        ge::DT_INT32,
                                                        ge::FORMAT_ND,
                                                        {-1},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  op.create_dynamic_output_y(3);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::cout << "ConcatOffset check" << std::endl;
  auto output_desc = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
}

TEST_F(ConcatOffset, concat_offset_infer_3) {
  ge::op::ConcatOffset op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{4, 18}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{3, 11}};
  std::vector<std::pair<int64_t,int64_t>> shape_range3 = {{2, 7}};
  auto tensor_desc = create_desc({4},
                                 ge::DT_INT32);
  auto tensor_desc_with_range1 = create_desc_shape_range({-1},
                                                        ge::DT_INT32,
                                                        ge::FORMAT_ND,
                                                        {-1},
                                                        ge::FORMAT_ND,
                                                        shape_range1);
  auto tensor_desc_with_range2 = create_desc_shape_range({-1},
                                                        ge::DT_INT32,
                                                        ge::FORMAT_ND,
                                                        {-1},
                                                        ge::FORMAT_ND,
                                                        shape_range2);
  auto tensor_desc_with_range3 = create_desc_shape_range({-1},
                                                        ge::DT_INT32,
                                                        ge::FORMAT_ND,
                                                        {-1},
                                                        ge::FORMAT_ND,
                                                        shape_range3);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_with_range1);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_with_range2);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range3);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  op.create_dynamic_output_y(3);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::cout << "ConcatOffset check" << std::endl;
  auto output_desc = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{4, 7}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

