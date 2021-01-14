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
 * @file test_BNInferenceD_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"

class BNInferenceD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNInferenceD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNInferenceD TearDown" << std::endl;
  }
};

TEST_F(BNInferenceD, BNInferenceD_infer_shape_fp16) {

  ge::op::BNInferenceD op;

  auto x_desc = create_desc_shape_range({-1,-1,-1,-1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,16,10,10}, ge::FORMAT_NCHW, 
    {{1, 1}, {-1, 16}, {10, 10}, {10, 10}});
  op.UpdateInputDesc("x", x_desc);

  auto mean_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{16, 16},});
  op.UpdateInputDesc("mean", mean_desc);

  auto variance_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{16, 16},});
  op.UpdateInputDesc("variance", variance_desc);

  auto scale_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{16, 16},});
  op.UpdateInputDesc("scale", scale_desc);

  auto offset_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {16,}, ge::FORMAT_ND, {{16, 16},});
  op.UpdateInputDesc("b", offset_desc);

  op.SetAttr("momentum", 0.999f);
  op.SetAttr("epsilon", 0.001f);
  op.SetAttr("use_global_stats", true);
  op.SetAttr("mode", 1);

  std::string data_format = "NCHW";
  op.SetAttr("data_format", data_format);

  // inference shape and shape range
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_y_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{{1, 1}, {-1, 16}, {10, 10}, {10, 10}}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
  
}
