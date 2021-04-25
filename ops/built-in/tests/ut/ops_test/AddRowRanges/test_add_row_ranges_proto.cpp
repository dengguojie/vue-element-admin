/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_add_row_ranges_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class add_row_ranges : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "add_row_ranges SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "add_row_ranges TearDown" << std::endl;
  }
};

TEST_F(add_row_ranges, add_row_ranges_infer_shape_fp32) {
  ge::op::AddRowRanges op;

  op.UpdateInputDesc("x", create_desc({16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("src", create_desc({16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16, 2}, ge::DT_INT32));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {16, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
