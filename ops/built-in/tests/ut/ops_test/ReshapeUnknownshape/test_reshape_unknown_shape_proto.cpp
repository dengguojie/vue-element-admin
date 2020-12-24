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
 * @file test_reshape_unknown_shape_proto.cpp
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


class RESHAPE_UNKNOWN_SHAPE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RESHAPE_UNKNOWN_SHAPE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RESHAPE_UNKNOWN_SHAPE_UT TearDown" << std::endl;
  }
};

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, InferShape) {
  ge::op::Reshape op("Reshape");
  op.UpdateInputDesc("x", create_desc({-1}, ge::DT_INT32));
  op.UpdateInputDesc("shape", create_desc({}, ge::DT_INT32));
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", -1);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape output_shape;
  bool ret = SetScalarOutputDesc(std::string("x"), std::string("y"), op_desc, output_shape);
  EXPECT_EQ(ret, true);
}