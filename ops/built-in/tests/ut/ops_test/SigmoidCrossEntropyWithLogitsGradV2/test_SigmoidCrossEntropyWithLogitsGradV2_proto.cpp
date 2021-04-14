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
 * @file test_AssignSub_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>  // NOLINT
#include "op_proto_test_util.h"  // NOLINT
#include "nn_norm_ops.h"  // NOLINT

class SigmoidCrossEntropyWithLogitsGradV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SigmoidCrossEntropyWithLogitsGradV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SigmoidCrossEntropyWithLogitsGradV2 TearDown" << std::endl;
  }
};

TEST_F(SigmoidCrossEntropyWithLogitsGradV2, scewlg_infer_shape_fp16) {
  ge::op::SigmoidCrossEntropyWithLogitsGradV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("predict", tensor_desc);
  op.UpdateInputDesc("target", tensor_desc);
  op.UpdateInputDesc("dout", tensor_desc);
  op.UpdateInputDesc("weight", tensor_desc);
  op.UpdateInputDesc("pos_weight", tensor_desc);
  op.SetAttr("reduction", "mean");


  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("gradient");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100}, };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
