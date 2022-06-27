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
 * @file test_mul_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
#include "common/utils/ut_op_common.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"


class drop_out_do_mask : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "drop_out_do_mask SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "drop_out_do_mask TearDown" << std::endl;
  }
};

TEST_F(drop_out_do_mask, drop_out_do_mask_1) {
  ge::op::DropOutDoMask op;
  op.UpdateInputDesc("x", create_desc_shape_range({6,7,2}, ge::DT_FLOAT, ge::FORMAT_ND, {6,7,2}, ge::FORMAT_ND, {{6,6}, {7,7}, {2,2}}));
  op.UpdateInputDesc("mask", create_desc_shape_range({1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {}));
  op.UpdateInputDesc("keep_prob", create_desc_shape_range({6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::FORMAT_ND, {}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {6,7,2};
  CommonInferShapeOperator(op, {}, {expected_output_shape});
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
