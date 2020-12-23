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

class StringFormat : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringFormat SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringFormat TearDown" << std::endl;
  }
};

TEST_F(StringFormat, string_format_infer_shape_with_normal) {
  ge::op::StringFormat op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{10, 10}, {10, 10}};
  auto tensor_desc = create_desc_shape_range({10, 10},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10, 10},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.SetAttr("template", "tensor: {}, suffix");
  op.SetAttr("placeholder", "{}");
  op.SetAttr("summarize", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(StringFormat, string_format_infer_shape_error) {
  
  ge::op::StringFormat op;
  op.create_dynamic_input_x(0);
  op.SetAttr("template", "tensor: {}, suffix");
  op.SetAttr("placeholder", "{}");
  op.SetAttr("summarize", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
