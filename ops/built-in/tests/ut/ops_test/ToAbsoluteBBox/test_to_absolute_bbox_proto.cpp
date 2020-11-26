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
 * @file test_to_absolute_bbox_proto.cpp
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

class to_absolute_bbox : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "to_absolute_bbox SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "to_absolute_bbox TearDown" << std::endl;
  }
};

TEST_F(to_absolute_bbox, to_absolute_bbox_infer_shape_fp16) {
  ge::op::ToAbsoluteBBox op;
  op.UpdateInputDesc("normalized_boxes", create_desc({2, 100, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("shape_hw", create_desc({4,}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(to_absolute_bbox, to_absolute_bbox_infer_shape_fp32) {
  ge::op::ToAbsoluteBBox op;
  op.UpdateInputDesc("normalized_boxes", create_desc({2, 100, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("shape_hw", create_desc({4,}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}



