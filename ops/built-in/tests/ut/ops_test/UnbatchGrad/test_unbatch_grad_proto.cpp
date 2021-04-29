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
 * @file test_unbatch_grad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "batch_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class UnbatchGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnbatchGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnbatchGrad TearDown" << std::endl;
  }
};

TEST_F(UnbatchGrad, unbatch_grad_infer_shape01) {
  ge::op::UnbatchGrad op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, dim_shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, dim_shape_range);
  op.UpdateInputDesc("x_input", tensor_desc1);
  op.UpdateInputDesc("grad", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UnbatchGrad, unbatch_grad_infer_shape02) {
  ge::op::UnbatchGrad op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, dim_shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, dim_shape_range);
  op.UpdateInputDesc("x_input", tensor_desc1);
  op.UpdateInputDesc("grad", tensor_desc2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}