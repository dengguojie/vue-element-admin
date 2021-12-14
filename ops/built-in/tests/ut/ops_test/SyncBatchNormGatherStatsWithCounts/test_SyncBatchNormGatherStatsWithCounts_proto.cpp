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
 * @file test_SyncBatchNormGatherStatsWithCounts_proto.cpp
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

class SyncBatchNormGatherStatsWithCountsTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sync_batch_norm_gather_stats_with_counts test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "sync_batch_norm_gather_stats_with_counts test TearDown" << std::endl;
    }
};

TEST_F(SyncBatchNormGatherStatsWithCountsTest, sync_batch_norm_gather_stats_with_countstest_case_1) {
  ge::op::SyncBatchNormGatherStatsWithCounts op;
  auto tensor_running_var_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1, 4},
                                             ge::FORMAT_ND, {{4, 4}});
  op.UpdateInputDesc("running_var", tensor_running_var_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc_invert_std_all = op.GetOutputDescByName("invert_std");
  EXPECT_EQ(output_desc_invert_std_all.GetDataType(), ge::DT_FLOAT);
  auto output_desc_running_var_update = op.GetOutputDescByName("running_var_update");
  EXPECT_EQ(output_desc_running_var_update.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = { -1 };
  EXPECT_EQ(output_desc_invert_std_all.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_running_var_update.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc_invert_std_all.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc_running_var_update.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}