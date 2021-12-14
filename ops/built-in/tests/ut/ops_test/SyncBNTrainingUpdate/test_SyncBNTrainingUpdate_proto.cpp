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
 * @file test_SyncBNTrainingUpdate_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"

class SyncBNTrainingUpdateTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sync_bn_training_update test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "sync_bn_training_update test TearDown" << std::endl;
    }
};

TEST_F(SyncBNTrainingUpdateTest, sync_bn_training_update_test_case_1) {
  ge::op::SyncBNTrainingUpdate op;
  auto tensor_mean_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1, 4},
                                             ge::FORMAT_ND, {{4, 4}});
  op.UpdateInputDesc("running_mean", tensor_mean_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc_running_mean_update = op.GetOutputDescByName("running_mean_update");
  EXPECT_EQ(output_desc_running_mean_update.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc_running_mean_update.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc_running_mean_update.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}