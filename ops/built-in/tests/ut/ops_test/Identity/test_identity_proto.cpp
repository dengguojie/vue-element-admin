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
 * @file test_identity_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>

#include "array_ops.h"

#include "op_proto_test_util.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"

class Identity : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Identity SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Identity TearDown" << std::endl;
  }
};

TEST_F(Identity, merge_infer_shape_known) {
  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);

  auto op = ge::op::Identity("identity");
  op.UpdateInputDesc("x", tensor_desc);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {32};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {};
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);
}

TEST_F(Identity, IdentityVauleShapeTest) {
  ge::op::Identity op("Identity");
  std::vector<std::pair<int64_t, int64_t>> x_range = {std::make_pair(1, 20), std::make_pair(1, 1),
                                                      std::make_pair(2, 2)};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->MutableInputDesc(0)->SetValueRange(x_range);
  auto ret = op_desc->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, 0);
  std::vector<std::pair<int64_t, int64_t>> y_value_range;
  op_desc->MutableOutputDesc(0)->GetValueRange(y_value_range);
  std::vector<int64_t> target_value_range = {1,20,1,1,2,2};
  std::vector<int64_t> output_value_range;
  for (auto pair : y_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  ASSERT_EQ(output_value_range, target_value_range);
}

TEST_F(Identity, IdentityNVauleShapeTest_failed) {
  ge::op::IdentityN op("IdentityN");

  std::vector<std::pair<int64_t, int64_t>> x_range = {std::make_pair(1, 20), std::make_pair(1, 1),
                                                      std::make_pair(2, 2)};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  vector<int64_t> dims_vec_0 = {8, 2};
  ge::GeTensorDesc tensor_desc_0(ge::GeShape(dims_vec_0), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  op_desc->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {3, 5, 4};
  ge::GeTensorDesc tensor_desc_1(ge::GeShape(dims_vec_1), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetRealDimCnt(tensor_desc_1, dims_vec_0.size());
  op_desc->AddInputDesc(tensor_desc_1);
  op_desc->MutableInputDesc(0)->SetValueRange(x_range);
  auto ret = op_desc->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, ge::GRAPH_PARAM_INVALID);
}
