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
 * @file test_SyncBatchNormBackwardReduce_proto.cpp
 */
 
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"

class SyncBatchNormBackwardReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sync_batch_norm_backward_reduce test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "sync_batch_norm_backward_reduce test TearDown" << std::endl;
    }
};

TEST_F(SyncBatchNormBackwardReduceTest, sync_batch_norm_backward_reduce_test_case_1) {
  ge::op::SyncBatchNormBackwardReduce op;
  auto tensor_sum_dy = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, {{1, 1}});
  auto tensor_sum_dy_dx_pad = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, {{1, 1}});
  auto tensor_mean = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, {{1, 1}});
  auto tensor_invert_std = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, {{1, 1}});
  op.UpdateInputDesc("sum_dy", tensor_sum_dy);
  op.UpdateInputDesc("sum_dy_dx_pad", tensor_sum_dy_dx_pad);
  op.UpdateInputDesc("mean", tensor_mean);
  op.UpdateInputDesc("invert_std", tensor_invert_std);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_sum_dy_xmu = op.GetOutputDescByName("sum_dy_xmu");
  EXPECT_EQ(output_desc_sum_dy_xmu.GetDataType(), ge::DT_FLOAT16);
  
  std::vector<int64_t> expected_output_shape = { -1 };
  EXPECT_EQ(output_desc_sum_dy_xmu.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc_sum_dy_xmu.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  auto output_desc_y = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
  
  std::vector<int64_t> expected_output_y_shape = { -1 };
  EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_y_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_y_shape_range;
  EXPECT_EQ(output_desc_y.GetShapeRange(output_y_shape_range), ge::GRAPH_SUCCESS);
}