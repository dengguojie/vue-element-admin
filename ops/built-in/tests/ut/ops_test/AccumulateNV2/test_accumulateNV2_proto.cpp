/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_accumulate_nv2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class accumulatenv2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "accumulatenv2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "accumulatenv2 TearDown" << std::endl;
  }
};

TEST_F(accumulatenv2, accumulatenv2_dynamic_infer_shape_fp16) {
  ge::op::AccumulateNV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2},
      {100, 200},
      {4, 8},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(accumulatenv2, accumulatenv2_static_infer_shape_fp16) {
  ge::op::AccumulateNV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{2, 2}, {100, 100}, {4, 4}};
  auto tensor_desc_1 = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range_1);
  std::vector<std::pair<int64_t,int64_t>> shape_range_2 = {{2, 2}, {100, 100}, {4, 4}};
  auto tensor_desc_2 = create_desc_shape_range({2, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range_2);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_1);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_2);
  op.SetAttr("N", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(accumulatenv2, accumulatenv2_static_same_shape_fp16) {
  ge::op::AccumulateNV2 op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 100, 4});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(accumulatenv2, accumulatenv2_static_diff_shape_fp16) {
  ge::op::AccumulateNV2 op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 100, 4});
  ge::Shape shape2({100, 4});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape); 

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  tensor_desc.SetShape(shape2);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(accumulatenv2, accumulatenv2_static_diff_shape_int8) {
  ge::op::AccumulateNV2 op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 100, 4});
  ge::Shape shape2({1});
  tensor_desc.SetDataType(ge::DT_INT8);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape); 

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  tensor_desc.SetShape(shape2);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(accumulatenv2, accumulatenv2_static_infer_shape_fp16_broadcast) {
  ge::op::AccumulateNV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{2, 2}, {100, 100}, {1, 1}};
  auto tensor_desc_1 = create_desc_shape_range({2, 100, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 1},
                                             ge::FORMAT_ND, shape_range_1);
  std::vector<std::pair<int64_t,int64_t>> shape_range_2 = {{2, 2}, {1, 1}, {4, 4}};
  auto tensor_desc_2 = create_desc_shape_range({2, 1, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 1, 4},
                                             ge::FORMAT_ND, shape_range_2);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_1);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_2);
  op.SetAttr("N", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
