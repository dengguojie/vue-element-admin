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
 * @file test_unique_with_counts_and_sorting_proto.cpp
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

class UniqueWithCountsAndSorting : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UniqueWithCountsAndSorting SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UniqueWithCountsAndSorting TearDown" << std::endl; 
  }
};

TEST_F(UniqueWithCountsAndSorting, UniqueWithCountsAndSortingInferShape1) {
  ge::op::UniqueWithCountsAndSorting op;
  op.UpdateInputDesc("x", create_desc({16, }, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y1");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(UniqueWithCountsAndSorting, UniqueWithCountsAndSortingInferShape2) {
  ge::op::UniqueWithCountsAndSorting op;
  op.UpdateInputDesc("x", create_desc({16, }, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y1");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}