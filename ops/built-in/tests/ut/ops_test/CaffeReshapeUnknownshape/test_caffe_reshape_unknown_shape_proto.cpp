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
 * @file test_caffe_reshape_unknown_shape_proto.cpp
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

class CAFFE_RESHAPE_UNKNOWN_SHAPE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CAFFE_RESHAPE_UNKNOWN_SHAPE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CAFFE_RESHAPE_UNKNOWN_SHAPE_UT TearDown" << std::endl;
  }
};

TEST_F(CAFFE_RESHAPE_UNKNOWN_SHAPE_UT, InferShape) {
  ge::op::Reshape op1;
  op1.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT32));
  op1.SetAttr("axis", 0);
  op1.SetAttr("num_axes", 1);

  std::vector<int64_t> attr_dims1 = {2,3,4};
  op1.SetAttr("shape", attr_dims1);
  op1.InferShapeAndType();

  auto y_desc1 = op1.GetOutputDesc("y");
  std::vector<int64_t> expected_y_shape1 = {-2};
  EXPECT_EQ(y_desc1.GetShape().GetDims(), expected_y_shape1);

  ge::op::Reshape op2;
  op2.UpdateInputDesc("x", create_desc({-1}, ge::DT_INT32));
  op2.SetAttr("axis", 0);
  op2.SetAttr("num_axes", 1);

  std::vector<int64_t> attr_dims2 = {2,3,4};
  op2.SetAttr("shape", attr_dims2);
  op2.InferShapeAndType();

  auto y_desc2 = op2.GetOutputDesc("y");
  std::vector<int64_t> expected_y_shape2 = {-1};
  EXPECT_EQ(y_desc2.GetShape().GetDims(), expected_y_shape2);


}