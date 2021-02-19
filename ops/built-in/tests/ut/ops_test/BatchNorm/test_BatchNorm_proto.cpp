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
 * @file test_BatchNorm_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"

class BatchNorm : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BatchNorm SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchNorm TearDown" << std::endl;
  }
};

TEST_F(BatchNorm, batchnorm_infer_shape_6d_fp16) {
  ge::op::BatchNorm op;
  auto tensor_desc_x = create_desc_shape_range({1, 64, 12, 12, 32, 32},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {1, 64, 12, 12, 32, 32},
                                                ge::FORMAT_ND, {{1, 1}, {64, 64}, {12, 12}, {12, 12}, {32, 32}, {32, 32}});
  auto tensor_desc_scale = create_desc_shape_range({64},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});     
  auto tensor_desc_offset = create_desc_shape_range({64},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                               
  auto tensor_desc_mean = create_desc_shape_range({64},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                               
  auto tensor_desc_variance = create_desc_shape_range({64},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                                                                                                                      
  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.UpdateInputDesc("offset", tensor_desc_offset); 
  op.UpdateInputDesc("mean", tensor_desc_mean);
  op.UpdateInputDesc("variance", tensor_desc_variance);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 64, 12, 12, 32, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(BatchNorm, batchnorm_infer_shape_4d_fp32) {
  ge::op::BatchNorm op;
  auto tensor_desc_x = create_desc_shape_range({2, 64, 224, 224},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {2, 64, 224, 224},
                                                ge::FORMAT_ND, {{2, 2}, {64, 64}, {224, 224}, {224, 224}});
  auto tensor_desc_scale = create_desc_shape_range({64},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});     
  auto tensor_desc_offset = create_desc_shape_range({64},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                               
  auto tensor_desc_mean = create_desc_shape_range({64},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                               
  auto tensor_desc_variance = create_desc_shape_range({64},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, {{64, 64}});                                                                                                                                      
  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.UpdateInputDesc("offset", tensor_desc_offset); 
  op.UpdateInputDesc("mean", tensor_desc_mean);
  op.UpdateInputDesc("variance", tensor_desc_variance);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 64, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}