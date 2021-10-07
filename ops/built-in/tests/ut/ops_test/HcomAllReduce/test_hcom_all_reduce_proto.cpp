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
 * @file test_hcom_all_reduce_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "hcom_ops.h"

class HcomAllReduceTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "HcomAllReduce SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HcomAllReduce TearDown" << std::endl;
  }
};

TEST_F(HcomAllReduceTest, hcom_all_reduce_infershape_test) {
  ge::op::HcomAllReduce op;
  op.UpdateInputDesc("x", create_desc({8, 1}, ge::DT_FLOAT));
  op.SetAttr("fusion", 2);
  op.SetAttr("fusionId", -1);
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("reduction", "sum");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.get_output_desc_y();
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {8, 1};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(HcomAllReduceTest, hcom_all_reduce_infershape_unknown_shape_test) {
  ge::op::HcomAllReduce op;
  op.UpdateInputDesc("x", create_desc({-1, 1}, ge::DT_FLOAT));
  op.SetAttr("fusion", 2);
  op.SetAttr("fusionId", -1);
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("reduction", "sum");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.get_output_desc_y();
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, 1};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(HcomAllReduceTest, hcom_all_reduce_invalid1_infershape_test) {
  ge::op::HcomAllReduce op;
  op.UpdateInputDesc("x", create_desc({8, 1}, ge::DT_FLOAT));
  op.SetAttr("fusion", 10);
  op.SetAttr("fusionId", -1);
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("reduction", "sum");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(HcomAllReduceTest, hcom_all_reduce_invalid2_infershape_test) {
  ge::op::HcomAllReduce op;
  op.UpdateInputDesc("x", create_desc({8, 1}, ge::DT_FLOAT));
  op.SetAttr("fusion", 2);
  op.SetAttr("fusionId", -10);
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("reduction", "sum");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(HcomAllReduceTest, hcom_all_reduce_invalid3_infershape_test) {
  ge::op::HcomAllReduce op;
  op.UpdateInputDesc("x", create_desc({8, 1}, ge::DT_FLOAT));
  op.SetAttr("fusion", 2);
  op.SetAttr("fusionId", -1);
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("reduction", "a");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}