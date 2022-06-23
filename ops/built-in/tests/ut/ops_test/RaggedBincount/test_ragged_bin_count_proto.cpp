/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file
 * except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0

 * RaggedBinCount ut case
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "graph/utils/op_desc_utils.h"
#include "all_ops.h"
#include "math_ops.h"
#include "common/utils/ut_op_util.h"

class RaggedBinCountTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedBinCountTest test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedBinCountTest test TearDown" << std::endl;
  }
};

using namespace ut_util;
TEST_F(RaggedBinCountTest, RaggedBinCountTest_input_ok_test) {
  using namespace ge;

  auto splits_shape = std::vector<int64_t>({6});
  auto splits_dtype = DT_INT64;

  auto values_shape = std::vector<int64_t>({10});
  auto values_dtype = DT_INT32;

  auto size_shape = std::vector<int64_t>({1});
  auto size_dtype = DT_INT32;

  std::vector<int32_t> size_data = {5};

  auto weights_shape = std::vector<int64_t>({10});
  auto weights_dtype = DT_FLOAT;

  std::vector<int64_t> expected_output_shape = {5, 5};

  auto test_op = op::RaggedBinCount("RaggedBinCount");
  TENSOR_INPUT_WITH_SHAPE(test_op, splits, splits_shape, splits_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, size_shape, size_dtype, FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  test_op.InferShapeAndType();
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_splits_dynamic_test) {
  using namespace ge;

  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{2, 10}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {10}, ge::FORMAT_ND, splits_range);

  auto values_shape = std::vector<int64_t>({10});
  auto values_dtype = DT_INT32;

  auto size_shape = std::vector<int64_t>({1});
  auto size_dtype = DT_INT32;

  std::vector<int32_t> size_data = {5};

  auto weights_shape = std::vector<int64_t>({10});
  auto weights_dtype = DT_FLOAT;

  auto test_op = op::RaggedBinCount("RaggedBinCount");
  test_op.UpdateInputDesc("splits", splits_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, size_shape, size_dtype, FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 9}, {5, 5}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_size_dynamic_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_2 = 2;  // size
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_shape = std::vector<int64_t>({-1});
  auto values_dtype = DT_INT32;

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_shape = std::vector<int64_t>({-1});
  auto weights_dtype = DT_FLOAT;

  test_op.UpdateInputDesc("splits", splits_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_values_1d_dynamic_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> values_range = {{1, 10}};
  auto values_desc = create_desc_shape_range({-1}, values_dtype, ge::FORMAT_ND, {10}, ge::FORMAT_ND, values_range);

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_shape = std::vector<int64_t>({10});
  auto weights_dtype = DT_FLOAT;

  test_op.UpdateInputDesc("splits", splits_desc);
  test_op.UpdateInputDesc("values", values_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {-1};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  std::vector<std::pair<int64_t, int64_t>> values_shape_range;
  EXPECT_EQ(input_values_desc.GetShapeRange(values_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_values_shape_range = {{1, 10}};
  EXPECT_EQ(values_shape_range, expected_values_shape_range);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {-1};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  std::vector<std::pair<int64_t, int64_t>> weights_shape_range;
  EXPECT_EQ(input_weights_desc.GetShapeRange(weights_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_weights_shape_range = {{1, 10}};
  EXPECT_EQ(weights_shape_range, expected_weights_shape_range);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_values_2d_dynamic_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> values_range = {{1, 10}, {5, 20}};
  auto values_desc =
      create_desc_shape_range({-1, -1}, values_dtype, ge::FORMAT_ND, {10, 20}, ge::FORMAT_ND, values_range);

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_shape = std::vector<int64_t>({10});
  auto weights_dtype = DT_FLOAT;

  test_op.UpdateInputDesc("splits", splits_desc);
  test_op.UpdateInputDesc("values", values_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {-1, -1};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  std::vector<std::pair<int64_t, int64_t>> values_shape_range;
  EXPECT_EQ(input_values_desc.GetShapeRange(values_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_values_shape_range = {{1, 10}, {5, 20}};
  EXPECT_EQ(values_shape_range, expected_values_shape_range);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {-1, -1};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  std::vector<std::pair<int64_t, int64_t>> weights_shape_range;
  EXPECT_EQ(input_weights_desc.GetShapeRange(weights_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_weights_shape_range = {{1, 10}, {5, 20}};
  EXPECT_EQ(weights_shape_range, expected_weights_shape_range);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_values_2d_dynamic_half_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> values_range = {{1, 10}};
  auto values_desc =
      create_desc_shape_range({-1, 20}, values_dtype, ge::FORMAT_ND, {10, 20}, ge::FORMAT_ND, values_range);

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_shape = std::vector<int64_t>({10});
  auto weights_dtype = DT_FLOAT;

  test_op.UpdateInputDesc("splits", splits_desc);
  test_op.UpdateInputDesc("values", values_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, weights, weights_shape, weights_dtype, FORMAT_ND, {});
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {-1, 20};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  std::vector<std::pair<int64_t, int64_t>> values_shape_range;
  EXPECT_EQ(input_values_desc.GetShapeRange(values_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_values_shape_range = {{1, 10}};
  EXPECT_EQ(values_shape_range, expected_values_shape_range);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {-1, 20};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  std::vector<std::pair<int64_t, int64_t>> weights_shape_range;
  EXPECT_EQ(input_weights_desc.GetShapeRange(weights_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_weights_shape_range = {{1, 10}};
  EXPECT_EQ(weights_shape_range, expected_weights_shape_range);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_weights_1d_dynamic_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_shape = std::vector<int64_t>({10});
  auto values_dtype = DT_INT32;

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_dtype = DT_FLOAT;
  std::vector<std::pair<int64_t, int64_t>> weights_range = {{5, 20}};
  auto weights_desc = create_desc_shape_range({-1}, weights_dtype, ge::FORMAT_ND, {20}, ge::FORMAT_ND, weights_range);

  test_op.UpdateInputDesc("splits", splits_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  test_op.UpdateInputDesc("weights", weights_desc);
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {10};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {10};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_weights_2d_dynamic_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_shape = std::vector<int64_t>({20});
  auto values_dtype = DT_INT32;

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_dtype = DT_FLOAT;
  std::vector<std::pair<int64_t, int64_t>> weights_range = {{5, 20}, {10, 30}};
  auto weights_desc =
      create_desc_shape_range({-1, -1}, weights_dtype, ge::FORMAT_ND, {20, 30}, ge::FORMAT_ND, weights_range);

  test_op.UpdateInputDesc("splits", splits_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  test_op.UpdateInputDesc("weights", weights_desc);
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {20};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {20};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(RaggedBinCountTest, RaggedBinCountTest_weights_empty_test) {
  using namespace ge;
  auto test_op = op::RaggedBinCount("RaggedBinCount");

  static const size_t input_idx_1 = 1;  // values
  static const size_t input_idx_2 = 2;  // size
  static const size_t input_idx_3 = 3;  // weights
  static const size_t output_idx_0 = 0;

  auto splits_dtype = DT_INT64;
  std::vector<std::pair<int64_t, int64_t>> splits_range = {{10, 100}};
  auto splits_desc = create_desc_shape_range({-1}, splits_dtype, ge::FORMAT_ND, {100}, ge::FORMAT_ND, splits_range);

  auto values_shape = std::vector<int64_t>({10});
  auto values_dtype = DT_INT32;

  auto size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> size_range = {{1, 1}};
  auto size_desc = create_desc_shape_range({-1}, size_dtype, ge::FORMAT_ND, {1}, ge::FORMAT_ND, size_range);
  test_op.UpdateInputDesc("size", size_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(test_op);
  auto input_size_desc = op_desc->MutableInputDesc(input_idx_2);
  std::vector<std::pair<int64_t, int64_t>> value_range = {{10, 80}};
  input_size_desc->SetValueRange(value_range);

  auto weights_dtype = DT_FLOAT;
  auto weights_desc = create_desc_shape_range({}, weights_dtype, ge::FORMAT_ND, {}, ge::FORMAT_ND, {});

  test_op.UpdateInputDesc("splits", splits_desc);
  TENSOR_INPUT_WITH_SHAPE(test_op, values, values_shape, values_dtype, FORMAT_ND, {});
  test_op.UpdateInputDesc("weights", weights_desc);
  test_op.SetAttr("binary_output", false);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = test_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto input_values_desc = test_op.GetInputDesc(input_idx_1);
  std::vector<int64_t> expected_values_shape = {10};
  EXPECT_EQ(input_values_desc.GetShape().GetDims(), expected_values_shape);
  auto input_weights_desc = test_op.GetInputDesc(input_idx_3);
  std::vector<int64_t> expected_weights_shape = {10};
  EXPECT_EQ(input_weights_desc.GetShape().GetDims(), expected_weights_shape);
  auto output_desc = test_op.GetOutputDesc(output_idx_0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> size_value_range;
  EXPECT_EQ(input_size_desc->GetValueRange(size_value_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_size_value_range = {{10, 80}};
  EXPECT_EQ(size_value_range, expected_size_value_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{9, 99}, {10, 80}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}