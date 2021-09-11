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
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

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

TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_fp1612) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 200}};
  auto tensor_desc = create_desc({-2,},
                                 ge::DT_FLOAT16);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
 // auto output_desc = op.GetOutputDesc("y");
 // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
 // std::vector<int64_t> expected_output_shape = {2, 100, 12};
 //// EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
 // std::vector<std::pair<int64_t,int64_t>> output_shape_range;
//  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
 // std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
 // EXPECT_EQ(output_shape_range, expected_shape_range);
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
TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_mix_fp1446) {
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

  auto ret = op.InferShapeAndType();
 
}
TEST_F(ConcatD, concat_d_infer_shape_no_shape_range_mix_fp1644) {
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

  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  
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

TEST_F(ConcatD, concat_d_infer_shape_dynamic3_fp16) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc =
      [&shape_range](std::initializer_list<int64_t> shape_dims) -> ge::TensorDesc {
        return create_desc_shape_range(shape_dims,
                                       ge::DT_FLOAT, ge::FORMAT_ND,
                                       {1, 12, -1, 64},
                                       ge::FORMAT_ND, shape_range);
      };

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc({1, 12, -1, 64}));
  op.UpdateDynamicInputDesc("x", 1, tensor_desc({1, 12, 1, 64}));
  op.SetAttr("N", 2);
  op.SetAttr("concat_dim", -2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 12, -1, 64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {1, 1},
      {12, 12},
      {2, -1},
      {64, 64},
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
      {3, -1},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ConcatD, concatd_data_slice_infer1) {
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

  std::vector<std::vector<int64_t>> y_data_slice ={{0,2}, {50,100}, {0, 12}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x0 = op_desc->MutableInputDesc("x0");
  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x0_data_slice;
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, x0_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> expected_x0_data_slice = {{0, 2}, {50, 100}, {0, 4}};
  std::vector<std::vector<int64_t>> expected_x1_data_slice = {{0, 2}, {50, 100}, {0, 4}};
  std::vector<std::vector<int64_t>> expected_x2_data_slice = {{0, 2}, {50, 100}, {0, 4}};
  EXPECT_EQ(expected_x0_data_slice, x0_data_slice);
  EXPECT_EQ(expected_x1_data_slice, x1_data_slice);
  EXPECT_EQ(expected_x2_data_slice, x2_data_slice);
}

TEST_F(ConcatD, concatd_data_slice_infer2) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto format = ge::FORMAT_ND;
  auto x0 =create_desc_shape_range({2, 100, 4}, ge::DT_FLOAT16, format, {2, 100, 4}, format, shape_range);
  auto x1 =create_desc_shape_range({2, 100, 5}, ge::DT_FLOAT16, format, {2, 100, 8}, format, shape_range);
  auto x2 =create_desc_shape_range({2, 100, 6}, ge::DT_FLOAT16, format, {2, 100, 5}, format, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, x0);
  op.UpdateDynamicInputDesc("x", 1, x1);
  op.UpdateDynamicInputDesc("x", 2, x2);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 15};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);

  std::vector<std::vector<int64_t>> y_data_slice ={{0,2}, {50,100}, {0, 15}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x0 = op_desc->MutableInputDesc("x0");
  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x0_data_slice;
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, x0_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> expected_x0_data_slice = {{0, 2}, {50, 100}, {0, 4}};
  std::vector<std::vector<int64_t>> expected_x1_data_slice = {{0, 2}, {50, 100}, {0, 5}};
  std::vector<std::vector<int64_t>> expected_x2_data_slice = {{0, 2}, {50, 100}, {0, 6}};
  EXPECT_EQ(expected_x0_data_slice, x0_data_slice);
  EXPECT_EQ(expected_x1_data_slice, x1_data_slice);
  EXPECT_EQ(expected_x2_data_slice, x2_data_slice);
}

TEST_F(ConcatD, concatd_data_slice_infer3) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {3, 3}, {100, 200}, {4, 8}, {16, 16}};
  auto format = ge::FORMAT_NC1HWC0;
  auto ori_format = ge::FORMAT_NCHW;
  auto x0 =create_desc_shape_and_origin_shape_range({2, 3, 100, 4, 16}, ge::DT_FLOAT16, format, {2, 48, 100, 4}, ori_format, shape_range);
  auto x1 =create_desc_shape_and_origin_shape_range({2, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 48, 100, 5}, ori_format, shape_range);
  auto x2 =create_desc_shape_and_origin_shape_range({2, 3, 100, 6, 16}, ge::DT_FLOAT16, format, {2, 48, 100, 6}, ori_format, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, x0);
  op.UpdateDynamicInputDesc("x", 1, x1);
  op.UpdateDynamicInputDesc("x", 2, x2);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto y = create_desc_shape_and_origin_shape_range({2, 3, 100, 15, 16}, ge::DT_FLOAT16, format, {2, 100, 15, 48}, ori_format, shape_range);
  op.UpdateOutputDesc("y", x0);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 15}, {0, 16}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x0 = op_desc->MutableInputDesc("x0");
  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x0_data_slice;
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, x0_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> expected_x0_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 4}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x1_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 5}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x2_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 6}, {0, 16}};
  EXPECT_EQ(expected_x0_data_slice, x0_data_slice);
  EXPECT_EQ(expected_x1_data_slice, x1_data_slice);
  EXPECT_EQ(expected_x2_data_slice, x2_data_slice);
}

TEST_F(ConcatD, concatd_data_slice_infer4) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {3, 3}, {100, 200}, {4, 8}, {16, 16}};
  auto format = ge::FORMAT_NC1HWC0;
  auto ori_format = ge::FORMAT_NHWC;
  auto x0 =create_desc_shape_and_origin_shape_range({2, 3, 100, 4, 16}, ge::DT_FLOAT16, format, {2, 100, 4, 48}, ori_format, shape_range);
  auto x1 =create_desc_shape_and_origin_shape_range({2, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 100, 5, 48}, ori_format, shape_range);
  auto x2 =create_desc_shape_and_origin_shape_range({2, 3, 100, 6, 16}, ge::DT_FLOAT16, format, {2, 100, 6, 48}, ori_format, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, x0);
  op.UpdateDynamicInputDesc("x", 1, x1);
  op.UpdateDynamicInputDesc("x", 2, x2);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -2);
  auto y = create_desc_shape_and_origin_shape_range({2, 3, 100, 15, 16}, ge::DT_FLOAT16, format, {2, 100, 15, 48}, ori_format, shape_range);
  op.UpdateOutputDesc("y", x0);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 15}, {0, 16}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x0 = op_desc->MutableInputDesc("x0");
  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x0_data_slice;
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, x0_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> expected_x0_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 4}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x1_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 5}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x2_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 6}, {0, 16}};
  EXPECT_EQ(expected_x0_data_slice, x0_data_slice);
  EXPECT_EQ(expected_x1_data_slice, x1_data_slice);
  EXPECT_EQ(expected_x2_data_slice, x2_data_slice);
}

TEST_F(ConcatD, concatd_data_slice_infer5) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {3, 3}, {100, 200}, {4, 8}, {16, 16}};
  auto format = ge::FORMAT_NC1HWC0;
  auto ori_format = ge::FORMAT_NHWC;
  auto x0 =create_desc_shape_and_origin_shape_range({2, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 100, 5, 48}, ori_format, shape_range);
  auto x1 =create_desc_shape_and_origin_shape_range({2, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 100, 5, 48}, ori_format, shape_range);
  auto x2 =create_desc_shape_and_origin_shape_range({2, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 100, 5, 48}, ori_format, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, x0);
  op.UpdateDynamicInputDesc("x", 1, x1);
  op.UpdateDynamicInputDesc("x", 2, x2);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto y = create_desc_shape_and_origin_shape_range({2, 9, 100, 15, 16}, ge::DT_FLOAT16, format, {2, 100, 15, 144}, ori_format, shape_range);
  op.UpdateOutputDesc("y", y);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {0, 9}, {50, 100}, {0, 5}, {0, 16}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x0 = op_desc->MutableInputDesc("x0");
  auto tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  auto tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x0_data_slice;
  std::vector<std::vector<int64_t>> x1_data_slice;
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, x0_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);
  std::vector<std::vector<int64_t>> expected_x0_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 5}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x1_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 5}, {0, 16}};
  std::vector<std::vector<int64_t>> expected_x2_data_slice = {{0, 2}, {0, 3}, {50, 100}, {0, 5}, {0, 16}};
  EXPECT_EQ(expected_x0_data_slice, x0_data_slice);
  EXPECT_EQ(expected_x1_data_slice, x1_data_slice);
  EXPECT_EQ(expected_x2_data_slice, x2_data_slice);
}

TEST_F(ConcatD, concatd_data_slice_infer6) {
  ge::op::ConcatD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {3, 3}, {100, 200}, {4, 8}, {16, 16}};
  auto format = ge::FORMAT_NDC1HWC0;
  auto ori_format = ge::FORMAT_NDHWC;
  auto x0 =create_desc_shape_range({2, 1, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 48}, ori_format, shape_range);
  auto x1 =create_desc_shape_range({2, 1, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 48}, ori_format, shape_range);
  auto x2 =create_desc_shape_range({2, 1, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 48}, ori_format, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, x0);
  op.UpdateDynamicInputDesc("x", 1, x1);
  op.UpdateDynamicInputDesc("x", 2, x2);
  op.SetAttr("N", 3);
  op.SetAttr("concat_dim", -1);
  auto y = create_desc_shape_range({2, 9, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 144}, ori_format, shape_range);
  op.UpdateOutputDesc("y", y);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {0, 1}, {0, 9}, {0, 100}, {0, 5}, {0, 16}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ConcatD, concat_infer_shape_fp161) {
  ge::op::Concat op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
