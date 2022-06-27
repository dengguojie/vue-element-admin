/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_gather_v2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "common/utils/ut_op_common.h"
class gather_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_v2 TearDown" << std::endl;
  }
};
using namespace ut_util;
TEST_F(gather_v2, gather_v2_infershape_runtime_test_1) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;
  // input axes info
  vector<int64_t> input_axes_shape = {1};
  auto axes_dtype = DT_INT32;

  vector<int32_t> axes_value = {2};
  // expect result info
  std::vector<int64_t> expected_output_shape ={6, 7,9, 10, 2};

  // gen GatherV2 op
  auto test_op = op::GatherV2("GatherV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, axis, input_axes_shape, axes_dtype, FORMAT_ND, axes_value);
  test_op.set_attr_batch_dims(0);

  // run InferShapeAndType
  test_op.InferShapeAndType();
  vector<bool> input_const = {false,false, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {"batch_dims"}, {expected_output_shape});
  Runtime2TestParam param;
  param.attrs = {"batch_dims"};
  param.input_const = {false,false, true};
  EXPECT_EQ(InferShapeTest(test_op, param), ge::GRAPH_SUCCESS);
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(gather_v2, gather_v2_infershape_runtime_test_2) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;
  // input axes info
  vector<int64_t> input_axes_shape = {1};
  auto axes_dtype = DT_INT64;

  vector<int64_t> axes_value = {2};
  // expect result info
  std::vector<int64_t> expected_output_shape ={6, 7,9, 10, 2};

  // gen GatherV2 op
  auto test_op = op::GatherV2("GatherV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, axis, input_axes_shape, axes_dtype, FORMAT_ND, axes_value);
  test_op.set_attr_batch_dims(0);

  // run InferShapeAndType
  test_op.InferShapeAndType();
  vector<bool> input_const = {false,false, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {"batch_dims"}, {expected_output_shape});
  Runtime2TestParam param;
  param.attrs = {"batch_dims"};
  param.input_const = {false,false, true};
  EXPECT_EQ(InferShapeTest(test_op, param), ge::GRAPH_SUCCESS);
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(gather_v2, gather_v2_infershape_runtime_test_3) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;
  // input axes info
  vector<int64_t> input_axes_shape = {1};
  auto axes_dtype = DT_INT32;

  vector<int32_t> axes_value = {-1};
  // expect result info
  std::vector<int64_t> expected_output_shape ={6, 7,9, 10, 2};

  // gen GatherV2 op
  auto test_op = op::GatherV2("GatherV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, axis, input_axes_shape, axes_dtype, FORMAT_ND, axes_value);
  test_op.set_attr_batch_dims(0);

  // run InferShapeAndType
  test_op.InferShapeAndType();
  vector<bool> input_const = {false,false, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {"batch_dims"}, {expected_output_shape});
  Runtime2TestParam param;
  param.attrs = {"batch_dims"};
  param.input_const = {false,false, true};
  EXPECT_EQ(InferShapeTest(test_op, param), ge::GRAPH_SUCCESS);
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(gather_v2, gather_v2_infershape_diff_test_1) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({6, 7}, ge::DT_INT32, ge::FORMAT_ND, {6, 7}, ge::FORMAT_ND, {{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({9, 10, 2}, ge::DT_INT32, ge::FORMAT_ND, {9, 10, 2}, ge::FORMAT_ND,{{9,9},{10,10},{2,2}}));
  /*ge::op::Constant axis;
  //int32_t value = 1;
  axis.SetAttr("value", std::vector<int32_t>{1});
  op.set_input_axis(axis);*/
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);


  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(2);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {6, 9, 10, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{6,6},{9,9},{10,10},{2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
  //delete []constData;
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_5) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,3},{4,5},{9,10}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND, {{10,10},{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  /*ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(3);
  op.UpdateInputDesc("x", tensor_x);*/

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_6) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {0};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_7) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1, 2}, ge::DT_INT32, ge::FORMAT_ND, {-1, 2}, ge::FORMAT_ND, {{3,4},{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3,4,-1,2,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{3,4},{2,2},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_8) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {0};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3,-1,-1,5,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{1,10},{4,4},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_10) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_11) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND,{{1,-1}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{0,-1},{0,-1},{0,-1},{0,-1},{0,-1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_with_batch_dims_1) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_input_axis(data0);

  op.SetAttr("batch_dims", 2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{{3,32},{3,32},{3,32},{3,32},{3,32}}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_with_batch_dims_2) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_input_axis(data0);

  op.SetAttr("batch_dims", 100);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  Runtime2TestParam param;
  param.attrs = {"batch_dims"};
  param.input_const = {false,false, true};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(gather_v2, gather_v2_infershape_with_batch_dims_3) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_input_axis(data0);

  // op.SetAttr("batch_dims", 2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{{3,32},{3,32},{3,32},{3,32},{3,32},{3,32},{3,32}}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_with_batch_dims_4) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_input_axis(data0);

  op.SetAttr("batch_dims", -100);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  Runtime2TestParam param;
  param.attrs = {"batch_dims"};
  param.input_const = {false,false, true};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(gather_v2, GatherV2_data_slice_infer1) {
  ge::op::GatherV2 op;

  auto tensor_desc = create_desc_with_ori({16, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 64}, ge::FORMAT_ND);
  op.UpdateInputDesc("indices", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);
  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  std::vector<std::vector<int64_t>> output_data_slice = {{10, 6}, {60, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(gather_v2, GatherV2_data_slice_infer2) {
  ge::op::GatherV2 op;

  auto tensor_desc = create_desc_with_ori({16, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 64}, ge::FORMAT_ND);
  op.UpdateInputDesc("indices", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);
  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  std::vector<std::vector<int64_t>> output_data_slice = {{10, 6}, {60, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
