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
 * @file test_addcmul_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class addcmul : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "addcmul test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "addcmul test TearDown" << std::endl;
  }
};

TEST_F(addcmul, addcmul_verify_test1_failed) {
  ge::op::Addcmul op;
  auto input_data_desc = create_desc_shape_range({-1, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 16, 1}, ge::FORMAT_NCHW,
   {{1, 3}, {1, 16}, {16, 16}, {1, 1}});
  op.UpdateInputDesc("input_data", input_data_desc);

  auto x1_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x1", x1_desc);

  auto x2_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x2", x2_desc);

  auto value_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16}});
  op.UpdateInputDesc("value", value_desc);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(addcmul, addcmul_verify_test1_success) {
  ge::op::Addcmul op;
  auto input_data_desc = create_desc_shape_range({-1, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 16, 1}, ge::FORMAT_NCHW,
   {{1, 3}, {1, 16}, {16, 16}, {1, 1}});
  op.UpdateInputDesc("input_data", input_data_desc);

  auto x1_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x1", x1_desc);

  auto x2_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x2", x2_desc);

  auto value_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16}});
  op.UpdateInputDesc("value", value_desc);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_SUCCESS);
}

TEST_F(addcmul, addcmul_infershape_fp16_test) {
  ge::op::Addcmul op;
  auto input_data_desc = create_desc_shape_range({-1, -1, -1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 16, 16, 1}, ge::FORMAT_NCHW,
   {{1, 3}, {1, 16}, {16, 16}, {1, 1}});
  op.UpdateInputDesc("input_data", input_data_desc);

  auto x1_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x1", x1_desc);

  auto x2_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16},});
  op.UpdateInputDesc("x2", x2_desc);

  auto value_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{1, 16}});
  op.UpdateInputDesc("value", value_desc);

  // inference shape and shape range
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_y_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{{1, 3}, {1, 16}, {16, 16}, {1, 16}}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
