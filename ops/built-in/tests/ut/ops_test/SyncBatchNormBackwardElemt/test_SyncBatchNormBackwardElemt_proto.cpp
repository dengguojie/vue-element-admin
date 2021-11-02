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
 * @file test_SyncBatchNormBackwardElemt_proto.cpp
 */
 
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"

class SyncBatchNormBackwardElemtTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sync_batch_norm_backward_reduce test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "sync_batch_norm_backward_reduce test TearDown" << std::endl;
    }
};

TEST_F(SyncBatchNormBackwardElemtTest, sync_batch_norm_backward_reduce_test_case_1) {
  ge::op::SyncBatchNormBackwardElemt op;
  auto tensor_grad_output = create_desc_shape_range({-1, 2, 4, 3},
                                                    ge::DT_FLOAT16, ge::FORMAT_ND,
                                                    {3, 2, 4, 3},
                                                    ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_save_input = create_desc_shape_range({-1, 2, 4, 3},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {3, 2, 4, 3},
                                                   ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_mean = create_desc_shape_range({-1, 2, 4, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {3, 2, 4, 3},
                                             ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_invstd = create_desc_shape_range({-1, 2, 4, 3},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {3, 2, 4, 3},
                                               ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_weight = create_desc_shape_range({-1, 2, 4, 3},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {3, 2, 4, 3},
                                               ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_mean_dy = create_desc_shape_range({-1, 2, 4, 3},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {3, 2, 4, 3},
                                                ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
  auto tensor_mean_dy_xmu = create_desc_shape_range({-1, 2, 4, 3},
                                                    ge::DT_FLOAT16, ge::FORMAT_ND,
                                                    {3, 2, 4, 3},
                                                    ge::FORMAT_ND, {{3, 3}, {2, 2}, {4, 4}, {3, 3}});
                                                    
  op.UpdateInputDesc("grad_output", tensor_grad_output);
  op.UpdateInputDesc("save_input", tensor_save_input);
  op.UpdateInputDesc("mean", tensor_mean);
  op.UpdateInputDesc("invstd", tensor_invstd);
  op.UpdateInputDesc("weight", tensor_weight);
  op.UpdateInputDesc("mean_dy", tensor_mean_dy);
  op.UpdateInputDesc("mean_dy_xmu", tensor_mean_dy_xmu);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_grad_input = op.GetOutputDescByName("grad_input");
  EXPECT_EQ(output_desc_grad_input.GetDataType(), ge::DT_FLOAT16);
  
  std::vector<int64_t> expected_output_y_shape = {-1, 2, 4, 3};
  EXPECT_EQ(output_desc_grad_input.GetShape().GetDims(), expected_output_y_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_y_shape_range;
  EXPECT_EQ(output_desc_grad_input.GetShapeRange(output_y_shape_range), ge::GRAPH_SUCCESS);
}