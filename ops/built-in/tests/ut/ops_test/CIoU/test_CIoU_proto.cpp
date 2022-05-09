/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_CIoU_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class CIoUTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "CIoUTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CIoUTest TearDown" << std::endl;
  }
};

TEST_F(CIoUTest, CIoUTest_infershape_test_1) {
  ge::op::CIoU op;
  op.UpdateInputDesc("bboxes", create_desc({4, 32}, ge::DT_FLOAT16));
  op.UpdateInputDesc("gtboxes", create_desc({4, 32}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_overlap_desc = op.GetOutputDescByName("overlap");
  EXPECT_EQ(output_overlap_desc.GetDataType(), ge::DT_FLOAT16);
  auto output_atan_sub_desc = op.GetOutputDescByName("atan_sub");
  EXPECT_EQ(output_atan_sub_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {1, 32};
  EXPECT_EQ(output_overlap_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_atan_sub_desc.GetShape().GetDims(), expected_output_shape);
}
