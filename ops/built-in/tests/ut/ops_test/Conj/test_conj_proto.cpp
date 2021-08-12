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
 * @file test_conj_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class Conj : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conj SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conj TearDown" << std::endl;
  }
};

TEST_F(Conj, Conj_infer_shape_0) {
  ge::op::Conj op;
  op.UpdateInputDesc("input", create_desc({-2}, ge::DT_COMPLEX64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
}

