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
#include "nn_training_ops.h"  // NOLINT

class ApplyAdagradD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdagradD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdagradD TearDown" << std::endl;
  }
};

TEST_F(ApplyAdagradD, scewlg_infer_shape_fp16) {
  ge::op::ApplyAdagradD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc_1 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, {{1, 2}});
  op.UpdateInputDesc("var", tensor_desc);
  op.UpdateInputDesc("accum", tensor_desc);
  op.UpdateInputDesc("lr", tensor_desc_1);
  op.UpdateInputDesc("grad", tensor_desc);
  op.SetAttr("update_slots", true);
  op.SetAttr("use_locking", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100}};
  EXPECT_EQ(output_shape_range, expected_shape_range);

  auto accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(accum_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_accum_output_shape = {-1};
  EXPECT_EQ(accum_desc.GetShape().GetDims(), expected_accum_output_shape);

  EXPECT_EQ(accum_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> accum_output_shape_range;
  EXPECT_EQ(accum_desc.GetShapeRange(accum_output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> accum_shape_range = {{2, 100}};
  EXPECT_EQ(accum_output_shape_range, accum_shape_range);
}
