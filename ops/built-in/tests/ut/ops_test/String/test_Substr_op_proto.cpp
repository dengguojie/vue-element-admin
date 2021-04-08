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
 * @file test_string_format_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "string_ops.h"

class Substr : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Substr SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Substr TearDown" << std::endl;
  }
};

TEST_F(Substr, substr_infer_shape) {
  ge::op::Substr op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("pos", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("len", create_desc({2}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Substr, substr_infer_shape_error) {
  ge::op::Substr op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("pos", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("len", create_desc({3}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Substr, substr_infer_shape_error1) {
  ge::op::Substr op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("pos", create_desc({2,1}, ge::DT_INT32));
  op.UpdateInputDesc("len", create_desc({3,1,2}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
