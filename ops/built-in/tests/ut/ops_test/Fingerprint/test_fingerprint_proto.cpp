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
 * @file test_Finferprint_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class Fingerprint : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Fingerprint SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Fingerprint TearDown" << std::endl;
  }
};

TEST_F(Fingerprint, finger_print_infer_shape_success) {
  ge::op::Fingerprint op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc0 = create_desc_shape_range({2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});
  op.UpdateInputDesc("data", tensor_desc0);
  op.UpdateInputDesc("method", tensor_desc1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Fingerprint, finger_print_infer_shape_fail1) {
  ge::op::Fingerprint op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc0 = create_desc_shape_range({2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("data", tensor_desc0);
  op.UpdateInputDesc("method", tensor_desc0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Fingerprint, finger_print_infer_shape) {
  ge::op::Fingerprint op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc0 = create_desc_shape_range({2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});
  op.UpdateInputDesc("data", tensor_desc1);
  op.UpdateInputDesc("method", tensor_desc1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}