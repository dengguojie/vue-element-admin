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
 * @file test_dynamic_AddN_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class AdamApplyOneAssign : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AdamApplyOneAssign SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AdamApplyOneAssign TearDown" << std::endl;
  }
};

TEST_F(AdamApplyOneAssign, adam_apply_one_case_0) {
  ge::op::AdamApplyOneAssign op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc_1 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, shape_range);      
  op.UpdateInputDesc("input0", tensor_desc);
  op.UpdateInputDesc("input1", tensor_desc);
  op.UpdateInputDesc("input2", tensor_desc);
  op.UpdateInputDesc("input3", tensor_desc);
  op.UpdateInputDesc("input4", tensor_desc_1);
  op.UpdateInputDesc("mul0_x", tensor_desc_1);
  op.UpdateInputDesc("mul1_x", tensor_desc_1);
  op.UpdateInputDesc("mul2_x", tensor_desc_1);
  op.UpdateInputDesc("mul3_x", tensor_desc_1);
  op.UpdateInputDesc("add2_y", tensor_desc_1);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}