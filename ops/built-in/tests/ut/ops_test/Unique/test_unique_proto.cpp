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
 * @file test_unique_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class UNIQUE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UNIQUE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UNIQUE_UT TearDown" << std::endl;
  }
};

TEST_F(UNIQUE_UT, InferShape) {
  ge::op::Unique op;
  op.UpdateInputDesc("x", create_desc({2, 16}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
}