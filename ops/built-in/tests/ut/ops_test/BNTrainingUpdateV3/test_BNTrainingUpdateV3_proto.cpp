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
 * @file test_BNTrainingUpdateV3_impl_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingUpdateV3 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingUpdateV3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingUpdateV3 TearDown" << std::endl;
  }
};

TEST_F(BNTrainingUpdateV3, bn_training_update_v3_test_1) {
  ge::op::BNTrainingUpdateV3 op;

  std::vector<std::pair<int64_t,int64_t>> shape_x_range = {{2,2}, {1,1000}, {1,1000}, {1,1000}, {16,16}};
  std::vector<std::pair<int64_t,int64_t>> shape_scale_range = {{1,1}, {1,1000}, {1,1000}, {1,1000}, {16,16}};

  auto tensor_desc_x = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_x_range);
  auto tensor_desc_sum = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_square_sum = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_scale = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_offset = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);


  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("sum", tensor_desc_sum);
  op.UpdateInputDesc("square_sum", tensor_desc_square_sum);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.UpdateInputDesc("offset", tensor_desc_offset);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  auto output_batch_mean_desc = op.GetOutputDesc("batch_mean");
  auto output_batch_variance_desc = op.GetOutputDesc("batch_variance");
  auto output_reserve_1_desc = op.GetOutputDesc("reserve_1");
  auto output_reserve_2_desc = op.GetOutputDesc("reserve_2");

  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_reserve_1_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_reserve_2_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, 16};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_batch_mean_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_batch_variance_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_reserve_1_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_reserve_2_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_y_shape_range;
  std::vector<std::pair<int64_t,int64_t>> output_batch_mean_shape_range;
  std::vector<std::pair<int64_t,int64_t>> output_batch_variance_shape_range;
  std::vector<std::pair<int64_t,int64_t>> output_reserve_1_shape_range;
  std::vector<std::pair<int64_t,int64_t>> output_reserve_2_shape_range;

  EXPECT_EQ(output_y_desc.GetShapeRange(output_y_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_batch_mean_desc.GetShapeRange(output_batch_mean_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_batch_variance_desc.GetShapeRange(output_batch_variance_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_reserve_1_desc.GetShapeRange(output_reserve_1_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_reserve_2_desc.GetShapeRange(output_reserve_2_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t,int64_t>> expected_y_shape_range = {
      {2, 2},
      {1, 1000},
      {1, 1000},
      {1, 1000},
      {16, 16}
    };
    std::vector<std::pair<int64_t,int64_t>> expected_scale_shape_range = {
      {1, 1},
      {1, 1000},
      {1, 1000},
      {1, 1000},
      {16, 16}
    };
  EXPECT_EQ(output_y_shape_range, expected_y_shape_range);
  EXPECT_EQ(output_batch_mean_shape_range, expected_scale_shape_range);
  EXPECT_EQ(output_batch_variance_shape_range, expected_scale_shape_range);
  EXPECT_EQ(output_reserve_1_shape_range, expected_scale_shape_range);
  EXPECT_EQ(output_reserve_2_shape_range, expected_scale_shape_range);
}

