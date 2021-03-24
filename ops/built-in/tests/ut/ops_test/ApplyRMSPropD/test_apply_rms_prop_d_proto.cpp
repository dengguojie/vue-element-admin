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
 * @file test_dynamic_AddN_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyRMSPropD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyRMSPropD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyRMSPropD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyRMSPropD, apply_rms_prop_d_infershape_verify_test) {
  ge::op::ApplyRMSPropD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc_1 = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("var", tensor_desc);
  op.UpdateInputDesc("ms", tensor_desc);
  op.UpdateInputDesc("mom", tensor_desc);
  op.UpdateInputDesc("lr", tensor_desc_1);
  op.UpdateInputDesc("grad", tensor_desc);
  op.SetAttr("rho", (float)0.5);
  op.SetAttr("momentum", (float)0.5);
  op.SetAttr("epsilon", (float)1.0e-7);

    auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_var_output_shape = {-1};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);
  EXPECT_EQ(var_desc.GetOriginShape().GetDims(), expected_var_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_var_shape_range;
  EXPECT_EQ(var_desc.GetShapeRange(output_var_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_var_shape_range = {{2, 100}};
  EXPECT_EQ(output_var_shape_range, expected_var_shape_range);

  auto ms_desc = op.GetOutputDesc("ms");
  EXPECT_EQ(ms_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_ms_output_shape = {-1};
  EXPECT_EQ(ms_desc.GetShape().GetDims(), expected_ms_output_shape);
  EXPECT_EQ(ms_desc.GetOriginShape().GetDims(), expected_ms_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_ms_shape_range;
  EXPECT_EQ(ms_desc.GetShapeRange(output_ms_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_ms_shape_range = {{2, 100}};
  EXPECT_EQ(output_ms_shape_range, expected_ms_shape_range);

  auto mom_desc = op.GetOutputDesc("mom");
  EXPECT_EQ(mom_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_mom_output_shape = {-1};
  EXPECT_EQ(mom_desc.GetShape().GetDims(), expected_mom_output_shape);
  EXPECT_EQ(mom_desc.GetOriginShape().GetDims(), expected_mom_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_mom_shape_range;
  EXPECT_EQ(mom_desc.GetShapeRange(output_mom_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_mom_shape_range = {{2, 100}};
  EXPECT_EQ(output_mom_shape_range, expected_mom_shape_range);
}
