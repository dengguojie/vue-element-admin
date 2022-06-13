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
 * @file test_gather_v2_d_proto.cpp
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
class gather_v2_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_v2_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_v2_d TearDown" << std::endl;
  }
};
using namespace ut_util;
TEST_F(gather_v2_d, gather_v2_d_infershape_runtime_test_1) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;

  // gen GatherV2D op
  auto test_op = op::GatherV2D("GatherV2D");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  test_op.set_attr_axis(2);

  // run InferShapeAndType
  test_op.InferShapeAndType();
}
TEST_F(gather_v2_d, gather_v2_d_infershape_runtime_test_2) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;

  // gen GatherV2D op
  auto test_op = op::GatherV2D("GatherV2D");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  test_op.set_attr_axis(0);

  // run InferShapeAndType
  test_op.InferShapeAndType();
}
TEST_F(gather_v2_d, gather_v2_d_infershape_runtime_test_3) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({6,7,2});
  auto input_x_dtype = DT_INT32;
  auto input_indices_shape = vector<int64_t>({9, 10, 2});
  auto input_indices_dtype = DT_INT32;

  // gen GatherV2D op
  auto test_op = op::GatherV2D("GatherV2D");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_indices_shape, input_indices_dtype, FORMAT_ND, {});
  test_op.set_attr_axis(-1);

  // run InferShapeAndType
  test_op.InferShapeAndType();
}
TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_1) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({6, 7}, ge::DT_INT32, ge::FORMAT_ND, {6, 7}, ge::FORMAT_ND, {{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({9, 10, 2}, ge::DT_INT32, ge::FORMAT_ND, {9, 10, 2}, ge::FORMAT_ND,{{9,9},{10,10},{2,2}}));
  op.set_attr_axis(1);


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
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{6,6},{9,9},{10,10},{2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
  //delete []constData;
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_5) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,3},{4,5},{9,10}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND, {{10,10},{2,2}}));
  op.set_attr_axis(2);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_6) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));
  op.set_attr_axis(0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_7) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1, 2}, ge::DT_INT32, ge::FORMAT_ND, {-1, 2}, ge::FORMAT_ND, {{3,4},{2,2}}));
  op.set_attr_axis(2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_8) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));
  op.set_attr_axis(0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_10) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));
  op.set_attr_axis(0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_diff_test_11) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND,{{1,-1}}));
  op.set_attr_axis(0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_with_batch_dims_1) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  op.set_attr_axis(2);
  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_with_batch_dims_2) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_attr_axis(2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_with_batch_dims_3) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  op.set_attr_axis(2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, gather_v2_d_infershape_with_batch_dims_4) {
  ge::op::GatherV2D op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  op.set_attr_axis(-100);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
}

TEST_F(gather_v2_d, GatherV2_data_slice_infer1) {
  ge::op::GatherV2D op;

  auto tensor_desc = create_desc_with_ori({16, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 64}, ge::FORMAT_ND);
  op.UpdateInputDesc("indices", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);
  op.set_attr_axis(0);

  std::vector<std::vector<int64_t>> output_data_slice = {{10, 6}, {60, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
}

TEST_F(gather_v2_d, GatherV2_data_slice_infer2) {
  ge::op::GatherV2D op;

  auto tensor_desc = create_desc_with_ori({16, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {16, 64}, ge::FORMAT_ND);
  op.UpdateInputDesc("indices", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateOutputDesc("y", tensor_desc);
  op.set_attr_axis(0);

  std::vector<std::vector<int64_t>> output_data_slice = {{10, 6}, {60, 4}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  auto status = op_desc->InferDataSlice();
}
