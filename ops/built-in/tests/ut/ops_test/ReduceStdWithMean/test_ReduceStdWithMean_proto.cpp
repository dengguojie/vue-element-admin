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
 * @file test_ReduceStd_proto.cpp
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
#include "reduce_ops.h"

class ReduceStdWithMeanTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "reduce_std_with_mean test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "reduce_std_with_mean test TearDown" << std::endl;
    }
};

TEST_F(ReduceStdWithMeanTest, reduce_std_with_meantest_case_1) {
  ge::op::ReduceStdWithMean op;
  auto tensor_x_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {3, 4},
                                             ge::FORMAT_ND, {{3, 3}, {4, 4}});
  auto tensor_mean_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {3, 1},
                                             ge::FORMAT_ND, {{3, 3}, {1, 1}});
  op.UpdateInputDesc("x", tensor_x_desc);
  op.UpdateInputDesc("mean", tensor_mean_desc);
  op.SetAttr("dim", {1,});
  op.SetAttr("unbiased", true);
  op.SetAttr("keep_dims", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc_y = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = { -1 };
  EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc_y.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}