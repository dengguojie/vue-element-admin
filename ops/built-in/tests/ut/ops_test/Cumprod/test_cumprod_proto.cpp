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
 * @file test_cumprod_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class Cumprod : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Cumprod SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Cumprod TearDown" << std::endl;
  }
};

TEST_F(Cumprod, Cumprod_infer_shape_0) {
  ge::op::Cumprod op;
  op.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
}

