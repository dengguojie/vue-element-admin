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
 * @file test_sparse_to_dense_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "sparse_ops.h"

class SPARSE_TO_DENSE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SPARSE_TO_DENSE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SPARSE_TO_DENSE_UT TearDown" << std::endl;
  }
};

TEST_F(SPARSE_TO_DENSE_UT, InferShape) {
  ge::op::SparseToDense op;
  op.UpdateInputDesc("indices", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("output_shape", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("default_value", create_desc({}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_y_shape = {-1, -1};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(SPARSE_TO_DENSE_UT, InferShape_dynamic) {
  ge::op::SparseToDense op;
  op.UpdateInputDesc("indices", create_desc({-1, 2}, ge::DT_INT32));
  op.UpdateInputDesc("output_shape", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("default_value", create_desc({}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
