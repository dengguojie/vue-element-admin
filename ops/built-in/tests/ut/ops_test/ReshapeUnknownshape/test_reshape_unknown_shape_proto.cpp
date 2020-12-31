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
#include <climits>
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

  int64_t a = 100;
  int64_t b = 10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = 100;
  b = -10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = -100;
  b = 10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = -100;
  b = -10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);
  std::vector<std::pair<int64_t, int64_t>> x_range1;
  int64_t range_max1 = 1;
  ge::array_ops::ReshapeRangeInfer(op, x_range1, range_max1);

  std::vector<std::pair<int64_t, int64_t>> x_range2;
  std::pair<int64_t,int64_t> pair1(16, 16);
  std::pair<int64_t,int64_t> pair2(1, -1);
  int64_t range_max2 = 1;
  x_range2.push_back(pair1);
  x_range2.push_back(pair2);
  ge::array_ops::ReshapeRangeInfer(op, x_range2, range_max2);
  
  std::vector<std::pair<int64_t, int64_t>> x_range3;
  std::vector<std::pair<int64_t, int64_t>> y_range3;
  std::pair<int64_t,int64_t> pair3(16, 16);
  std::pair<int64_t,int64_t> pair4(1, -1);
  x_range3.push_back(pair3);
  x_range3.push_back(pair4);
  std::vector<int64_t> dims1{4,-1};
  ge::GeShape shape3(dims1);
  ge::array_ops::ReshapeRangeInfer(op, x_range3, y_range3, shape3);
  
  std::vector<std::pair<int64_t, int64_t>> x_range4;
  std::vector<std::pair<int64_t, int64_t>> y_range4;
  std::pair<int64_t,int64_t> pair5(16, 16);
  x_range4.push_back(pair5);
  std::vector<int64_t> dims2{4, -1};
  ge::GeShape shape4(dims2);
  ge::array_ops::ReshapeRangeInfer(op, x_range4, y_range4, shape4);  
  EXPECT_EQ(ret, true);
}