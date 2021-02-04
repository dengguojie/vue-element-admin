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

class ConcatD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatD TearDown" << std::endl;
  }
};

TEST_F(ConcatD, concat_d_infer_shape_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_mix_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{2, 2}, {100, 100}, {12, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_mix_fp16_fail) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, create_desc({2, 101, 8},
                                                ge::DT_FLOAT16));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ConcatD, concat_d_infer_shape_dynamic_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_d_infer_shape_dynamic2_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, 5}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_fp16) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_no_shape_range_fp16) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_no_shape_range_mix_fp16) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{2, 2}, {100, 100}, {12, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_no_shape_range_mix_fp16_fail) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, create_desc({2, 101, 8},
                                                ge::DT_FLOAT16));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_dynamic_fp16) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_v2_d_infer_shape_dynamic2_fp16) {
  ge::op::ConcatV2D op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, 5}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_fp16) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_no_shape_range_fp16) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_no_shape_range_mix_fp16) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
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
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{2, 2}, {100, 100}, {12, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_unknow_concat_dim) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({4, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);

  auto concat_dim_desc = create_desc({1},
                                     ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", concat_dim_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{4, 10}, {100, 400}, {4, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_unknow_concat_dim) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({4, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_with_range);
  op.SetAttr("N", 3);

  auto concat_dim_desc = create_desc({1},
                                     ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", concat_dim_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{4, 10}, {100, 400}, {4, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_no_shape_range_mix_fp16_fail) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, create_desc({2, 101, 8},
                                                ge::DT_FLOAT16));
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
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ConcatD, concat_infer_shape_dynamic_fp16) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concat_infer_shape_dynamic2_fp16) {
  ge::op::Concat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, 5}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_no_shape_range_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 12};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_no_shape_range_mix_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
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
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{2, 2}, {100, 100}, {12, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_no_shape_range_mix_fp16_fail) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc({2, 100, 4},
                                 ge::DT_FLOAT16);
  auto tensor_desc_with_range = create_desc_shape_range({-1, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_ND,
                                                        {2, 100, 4},
                                                        ge::FORMAT_ND,
                                                        shape_range);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, create_desc({2, 101, 8},
                                                ge::DT_FLOAT16));
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
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic2_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, 5}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 3);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {12, 24},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic3_fp16) {
  ge::op::ConcatV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({16000, 30, 80}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({16000, 9, -1}));
  op.SetAttr("N", 2);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_concat_dim(concat_dim);
  auto desc = op.GetInputDesc("concat_dim");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("concat_dim", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16000, 39, 80};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic4_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, -1}, {4, -1}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, 5}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, -1},
      {12, -1},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic5_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, -1}, {4, -1}};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2, 100, 4},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1, -1, -1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1, -1, -1}));
  op.SetAttr("N", 4);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ConcatD, concatv2_infer_shape_dynamic6_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT16, ge::FORMAT_ND,
                                       {2},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({-1}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({-1}));
  op.UpdateDynamicInputDesc("x", 2, tensor_desc({-1}));
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {1, -1},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
