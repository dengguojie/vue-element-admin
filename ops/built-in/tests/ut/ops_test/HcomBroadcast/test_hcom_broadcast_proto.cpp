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
 * @file test_hcom_broadcast_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "op_proto_test_util.h"
#include "hcom_ops.h"

class HcomBroadcastTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "HcomBroadcast SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HcomBroadcast TearDown" << std::endl;
  }
};

TEST_F(HcomBroadcastTest, hcom_broadcast_infershape_test) {
  ge::op::HcomBroadcast op;

  op.create_dynamic_input_x(1);
  op.create_dynamic_output_y(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc({8, 1}, ge::DT_FLOAT));
  op.SetAttr("root_rank", 0);
  op.SetAttr("group", "hccl_world_group");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto outputDesc = opDesc->GetOutputDesc(0);

  EXPECT_EQ(outputDesc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {8, 1};
  EXPECT_EQ(outputDesc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(HcomBroadcastTest, hcom_broadcast_infershape_unknown_shape_test) {
  ge::op::HcomBroadcast op;

  op.create_dynamic_input_x(1);
  op.create_dynamic_output_y(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc({-1, 1}, ge::DT_FLOAT));
  op.SetAttr("root_rank", 0);
  op.SetAttr("group", "hccl_world_group");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto outputDesc = opDesc->GetOutputDesc(0);

  EXPECT_EQ(outputDesc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, 1};
  EXPECT_EQ(outputDesc.GetShape().GetDims(), expected_output_shape);
}