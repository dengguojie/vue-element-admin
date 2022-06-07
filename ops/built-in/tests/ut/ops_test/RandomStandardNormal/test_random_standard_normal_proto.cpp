/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_random_standard_normal_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"

class RandomStandardNormal : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RandomStandardNormal SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RandomStandardNormal TearDown" << std::endl;
  }
};

TEST_F(RandomStandardNormal, random_standard_normal_infer_shape) {
  ge::op::RandomStandardNormal op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("shape", tensor_desc);
  op.SetAttr("dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(RandomStandardNormal, random_standard_normal_infer_shape_failed) {
  ge::op::RandomStandardNormal op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("shape", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}