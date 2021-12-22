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
 * @file test_LoopCond_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>

#include "functional_ops.h"
#include "control_flow_ops.h"

#include "op_proto_test_util.h"
#include "graph/utils/graph_utils.h"
#include "utils/op_desc_utils.h"

class LoopCond : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LoopCond SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LoopCond TearDown" << std::endl;
  }
};

TEST_F(LoopCond, LoopCond_infer_shape_fail) {
  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);

  auto op = ge::op::LoopCond("loopcond");
  op.UpdateInputDesc("x", tensor_desc);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED);
}