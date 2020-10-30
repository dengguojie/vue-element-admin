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
 * @file test_strided_slice_proto.cpp
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
#include "selection_ops.h"

class strided_slice : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice TearDown" << std::endl;
  }
};

TEST_F(strided_slice, strided_slice_infer_shape_fp16) {
  ge::op::StridedSlice op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {38},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  
  ge::Tensor constTensorBegin;
  ge::TensorDesc constDescBegin(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescBegin.SetSize(1 * sizeof(int32_t));
  constTensorBegin.SetTensorDesc(constDescBegin);
  int32_t constDataBegin[1] = {0};
  constTensorBegin.SetData((uint8_t*)constDataBegin, 1 * sizeof(int32_t));
  auto begin = ge::op::Constant().set_attr_value(constTensorBegin);
  op.set_input_begin(begin);
  auto descBegin = op.GetInputDesc("begin");
  descBegin.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("begin", descBegin);

  ge::Tensor constTensorEnd;
  ge::TensorDesc constDescEnd(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescEnd.SetSize(1 * sizeof(int32_t));
  constTensorEnd.SetTensorDesc(constDescEnd);
  int32_t constDataEnd[1] = {36};
  constTensorEnd.SetData((uint8_t*)constDataEnd, 1 * sizeof(int32_t));
  auto end = ge::op::Constant().set_attr_value(constTensorEnd);
  op.set_input_end(end);
  auto descEnd = op.GetInputDesc("end");
  descEnd.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("end", descEnd);

  ge::Tensor constTensorStrides;
  ge::TensorDesc constDescStrides(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescStrides.SetSize(1 * sizeof(int32_t));
  constTensorStrides.SetTensorDesc(constDescStrides);
  int32_t constDataStrides[1] = {1};
  constTensorStrides.SetData((uint8_t*)constDataStrides, 1 * sizeof(int32_t));
  auto strides = ge::op::Constant().set_attr_value(constTensorStrides);
  op.set_input_strides(strides);
  auto descStrides = op.GetInputDesc("strides");
  descStrides.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("strides", descStrides);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 36},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape2_fp16) {
  ge::op::StridedSlice op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({16, -1, -1, -1, 128, 2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, -1, -1, -1, 128, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  ge::Tensor constTensorBegin;
  ge::TensorDesc constDescBegin(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescBegin.SetSize(2 * sizeof(int32_t));
  constTensorBegin.SetTensorDesc(constDescBegin);
  int32_t constDataBegin[2] = {0, 0};
  constTensorBegin.SetData((uint8_t*)constDataBegin, 2 * sizeof(int32_t));
  auto begin = ge::op::Constant().set_attr_value(constTensorBegin);
  op.set_input_begin(begin);
  auto descBegin = op.GetInputDesc("begin");
  descBegin.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("begin", descBegin);

  ge::Tensor constTensorEnd;
  ge::TensorDesc constDescEnd(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescEnd.SetSize(2 * sizeof(int32_t));
  constTensorEnd.SetTensorDesc(constDescEnd);
  int32_t constDataEnd[2] = {0, 1,};
  constTensorEnd.SetData((uint8_t*)constDataEnd, 2 * sizeof(int32_t));
  auto end = ge::op::Constant().set_attr_value(constTensorEnd);
  op.set_input_end(end);
  auto descEnd = op.GetInputDesc("end");
  descEnd.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("end", descEnd);

  ge::Tensor constTensorStrides;
  ge::TensorDesc constDescStrides(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescStrides.SetSize(2 * sizeof(int32_t));
  constTensorStrides.SetTensorDesc(constDescStrides);
  int32_t constDataStrides[2] = {1, 1};
  constTensorStrides.SetData((uint8_t*)constDataStrides, 2 * sizeof(int32_t));
  auto strides = ge::op::Constant().set_attr_value(constTensorStrides);
  op.set_input_strides(strides);
  auto descStrides = op.GetInputDesc("strides");
  descStrides.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("strides", descStrides);

  op.SetAttr("begin_mask", 0);
  op.SetAttr("ellipsis_mask", 1);
  op.SetAttr("end_mask", 0);
  op.SetAttr("new_axis_mask", 0);
  op.SetAttr("shrink_axis_mask", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16, -1, -1, -1, 128};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(strided_slice, strided_slice_infer_shape3_fp16) {
  ge::op::StridedSlice op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({16, 76, 76, 3, 128, 2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 76, 76, 3, 128, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  ge::Tensor constTensorBegin;
  ge::TensorDesc constDescBegin(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescBegin.SetSize(2 * sizeof(int32_t));
  constTensorBegin.SetTensorDesc(constDescBegin);
  int32_t constDataBegin[2] = {0, 0};
  constTensorBegin.SetData((uint8_t*)constDataBegin, 2 * sizeof(int32_t));
  auto begin = ge::op::Constant().set_attr_value(constTensorBegin);
  op.set_input_begin(begin);
  auto descBegin = op.GetInputDesc("begin");
  descBegin.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("begin", descBegin);

  ge::Tensor constTensorEnd;
  ge::TensorDesc constDescEnd(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescEnd.SetSize(2 * sizeof(int32_t));
  constTensorEnd.SetTensorDesc(constDescEnd);
  int32_t constDataEnd[2] = {0, 1,};
  constTensorEnd.SetData((uint8_t*)constDataEnd, 2 * sizeof(int32_t));
  auto end = ge::op::Constant().set_attr_value(constTensorEnd);
  op.set_input_end(end);
  auto descEnd = op.GetInputDesc("end");
  descEnd.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("end", descEnd);

  ge::Tensor constTensorStrides;
  ge::TensorDesc constDescStrides(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescStrides.SetSize(2 * sizeof(int32_t));
  constTensorStrides.SetTensorDesc(constDescStrides);
  int32_t constDataStrides[2] = {1, 1};
  constTensorStrides.SetData((uint8_t*)constDataStrides, 2 * sizeof(int32_t));
  auto strides = ge::op::Constant().set_attr_value(constTensorStrides);
  op.set_input_strides(strides);
  auto descStrides = op.GetInputDesc("strides");
  descStrides.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("strides", descStrides);

  op.SetAttr("begin_mask", 0);
  op.SetAttr("ellipsis_mask", 1);
  op.SetAttr("end_mask", 0);
  op.SetAttr("new_axis_mask", 0);
  op.SetAttr("shrink_axis_mask", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16, 76, 76, 3, 128};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(strided_slice, strided_slice_infer_shape4_fp16) {
  ge::op::StridedSlice op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1,13,13,3,2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10,13,13,3,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  ge::Tensor constTensorBegin;
  ge::TensorDesc constDescBegin(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescBegin.SetSize(2 * sizeof(int32_t));
  constTensorBegin.SetTensorDesc(constDescBegin);
  int32_t constDataBegin[2] = {0, 0};
  constTensorBegin.SetData((uint8_t*)constDataBegin, 2 * sizeof(int32_t));
  auto begin = ge::op::Constant().set_attr_value(constTensorBegin);
  op.set_input_begin(begin);
  auto descBegin = op.GetInputDesc("begin");
  descBegin.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("begin", descBegin);

  ge::Tensor constTensorEnd;
  ge::TensorDesc constDescEnd(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescEnd.SetSize(2 * sizeof(int32_t));
  constTensorEnd.SetTensorDesc(constDescEnd);
  int32_t constDataEnd[2] = {1, 10,};
  constTensorEnd.SetData((uint8_t*)constDataEnd, 2 * sizeof(int32_t));
  auto end = ge::op::Constant().set_attr_value(constTensorEnd);
  op.set_input_end(end);
  auto descEnd = op.GetInputDesc("end");
  descEnd.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("end", descEnd);

  ge::Tensor constTensorStrides;
  ge::TensorDesc constDescStrides(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescStrides.SetSize(2 * sizeof(int32_t));
  constTensorStrides.SetTensorDesc(constDescStrides);
  int32_t constDataStrides[2] = {1, 1};
  constTensorStrides.SetData((uint8_t*)constDataStrides, 2 * sizeof(int32_t));
  auto strides = ge::op::Constant().set_attr_value(constTensorStrides);
  op.set_input_strides(strides);
  auto descStrides = op.GetInputDesc("strides");
  descStrides.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("strides", descStrides);

  op.SetAttr("begin_mask", 0);
  op.SetAttr("ellipsis_mask", 0);
  op.SetAttr("end_mask", 0);
  op.SetAttr("new_axis_mask", 1);
  op.SetAttr("shrink_axis_mask", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, -1, 13, 13, 3, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

