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
 * @file test_matrixDiagD_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"
#include "split_combination_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "array_ops.h"

class MatrixDiagD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MatrixDiagD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MatrixDiagD TearDown" << std::endl;
  }
};

TEST_F(MatrixDiagD, MatrixDiagDInferShape) {
  ge::op::MatrixDiagD op;
  op.UpdateInputDesc("X", create_desc_with_ori({2,4}, ge::DT_INT32, ge::FORMAT_NHWC,{2,4}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("assist", create_desc_with_ori({2,4,4}, ge::DT_INT32, ge::FORMAT_NHWC,{2,4,4}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({2,4}, ge::DT_INT32, ge::FORMAT_NHWC,{2,4}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
