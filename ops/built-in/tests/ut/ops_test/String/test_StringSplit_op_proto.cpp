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

class StringSplit : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringSplit SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringSplit TearDown" << std::endl;
  }
};

TEST_F(StringSplit, string_split_infer_shape) {
  ge::op::StringSplit op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("delimiter", create_desc({}, ge::DT_STRING));
  op.SetAttr("skip_empty", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringSplit, string_split_infer_shape_error1) {
  ge::op::StringSplit op;
  op.UpdateInputDesc("input", create_desc({2,1}, ge::DT_STRING));
  op.UpdateInputDesc("delimiter", create_desc({}, ge::DT_STRING));
  op.SetAttr("skip_empty", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringSplit, string_split_infer_shape_error2) {
  ge::op::StringSplit op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("delimiter", create_desc({1}, ge::DT_STRING));
  op.SetAttr("skip_empty", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringSplit, string_split_v2_infer_shape) {
  ge::op::StringSplitV2 op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("sep", create_desc({}, ge::DT_STRING));
  op.SetAttr("maxsplit", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringSplit, string_split_v2_infer_shape_error1) {
  ge::op::StringSplitV2 op;
  op.UpdateInputDesc("input", create_desc({2,1}, ge::DT_STRING));
  op.UpdateInputDesc("sep", create_desc({}, ge::DT_STRING));
  op.SetAttr("maxsplit", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringSplit, string_split_v2_infer_shape_error2) {
  ge::op::StringSplitV2 op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("sep", create_desc({1}, ge::DT_STRING));
  op.SetAttr("maxsplit", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
