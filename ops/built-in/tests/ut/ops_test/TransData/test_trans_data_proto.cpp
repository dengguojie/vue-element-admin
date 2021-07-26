/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_trans_data_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "transformation_ops.h"

class trans_data : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "trans_data SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "trans_data TearDown" << std::endl;
  }
};

TEST_F(trans_data, trans_data_infer_shape_fp16) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto tensor_desc =
      create_desc_shape_range({16, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {64}, ge::FORMAT_NCHW, shape_range);
  auto tensor_desc_out = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW, shape_range);
  op.UpdateInputDesc("src", tensor_desc);
  op.UpdateOutputDesc("dst", tensor_desc_out);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("dst");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {16, 16, 16, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(trans_data, trans_data_infer_shape_with_diff_format) {
  ge::op::TransData op;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto tensor_desc =
      create_desc_shape_range({16, 16, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {64}, ge::FORMAT_NCHW, shape_range);
  auto tensor_desc_out = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("src", tensor_desc);
  op.UpdateOutputDesc("dst", tensor_desc_out);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("dst");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
