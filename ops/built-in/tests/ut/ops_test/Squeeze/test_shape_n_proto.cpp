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
 * @file test_shape_n_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

class shape_n_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "shape_n_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "shape_n_ut TearDown" << std::endl;
  }
};

TEST_F(shape_n_ut, shape_n_success) {
  ge::op::ShapeN op("ShapeN");
  op.create_dynamic_input_x(1);
  op.SetAttr("dtype",ge::DT_INT32);
  op.UpdateDynamicInputDesc("x", 0, create_desc({1, 2}, ge::DT_INT32));
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(shape_n_ut, shape_n_unknown_rank) {
  ge::op::ShapeN op("ShapeN");
  op.create_dynamic_input_x(1);
  op.SetAttr("dtype",ge::DT_INT32);
  op.UpdateDynamicInputDesc("x", 0, create_desc({-2, -1}, ge::DT_INT32));
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}