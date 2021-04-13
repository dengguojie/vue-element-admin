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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "set_ops.h"

class DenseToDenseSetOperationTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DenseToDenseSetOperationTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DenseToDenseSetOperationTest TearDown" << std::endl;
  }
};

TEST_F(DenseToDenseSetOperationTest, InferShape_01) {
  ge::op::DenseToDenseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto indices_desc = op.GetOutputDesc("y_indices");
  EXPECT_EQ(indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_indices_shape = {-1, 2};
  EXPECT_EQ(indices_desc.GetShape().GetDims(), expected_indices_shape);

  auto values_desc = op.GetOutputDesc("y_values");
  EXPECT_EQ(values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_values_shape = {-1};
  EXPECT_EQ(values_desc.GetShape().GetDims(), expected_values_shape);

  auto shape_desc = op.GetOutputDesc("y_shape");
  EXPECT_EQ(shape_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_shape = {2};
  EXPECT_EQ(shape_desc.GetShape().GetDims(), expected_shape);
}

//error input num
TEST_F(DenseToDenseSetOperationTest, InferShape_02) {
  ge::op::DenseToDenseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//erro x1 rank
TEST_F(DenseToDenseSetOperationTest, InferShape_03) {
  ge::op::DenseToDenseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error x2 rank
TEST_F(DenseToDenseSetOperationTest, InferShape_04) {
  ge::op::DenseToDenseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error data type
TEST_F(DenseToDenseSetOperationTest, InferShape_05) {
  ge::op::DenseToDenseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("x2", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// x1 unknow rank, error x2 rank
TEST_F(DenseToDenseSetOperationTest, InferShape_06) {
  ge::op::DenseToDenseSetOperation op;

  ge::TensorDesc x1_desc;
  x1_desc.SetDataType(ge::DT_FLOAT);
  ge::Shape x1_shape(ge::UNKNOWN_RANK);
  x1_desc.SetShape(x1_shape);
  op.UpdateInputDesc("x1", x1_desc);
  op.UpdateInputDesc("x2", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//x1 unknow rank, x2 unknow rank
TEST_F(DenseToDenseSetOperationTest, InferShape_07) {
  ge::op::DenseToDenseSetOperation op;

  ge::TensorDesc x1_desc;
  x1_desc.SetDataType(ge::DT_FLOAT);
  ge::Shape x1_shape(ge::UNKNOWN_RANK);
  x1_desc.SetShape(x1_shape);
  op.UpdateInputDesc("x1", x1_desc);

  ge::TensorDesc x2_desc;
  x2_desc.SetDataType(ge::DT_INT64);
  ge::Shape x2_shape(ge::UNKNOWN_RANK);
  x2_desc.SetShape(x2_shape);
  op.UpdateInputDesc("x2", x2_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//x1 unknow rank
TEST_F(DenseToDenseSetOperationTest, InferShape_08) {
  ge::op::DenseToDenseSetOperation op;
  ge::TensorDesc x1_desc;
  x1_desc.SetDataType(ge::DT_FLOAT);
  ge::Shape x1_shape(ge::UNKNOWN_RANK);
  x1_desc.SetShape(x1_shape);
  op.UpdateInputDesc("x1", x1_desc);
  op.UpdateInputDesc("x2", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}