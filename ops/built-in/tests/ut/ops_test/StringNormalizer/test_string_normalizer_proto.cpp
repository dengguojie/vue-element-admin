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
 * @file test_string_normalizer_proto.cpp
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

class StringNormalizer : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringNormalizer SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringNormalizer TearDown" << std::endl;
  }
};

TEST_F(StringNormalizer, string_normalizer_verify_failed1){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({3}, ge::DT_INT64));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringNormalizer, string_normalizer_verify_failed2){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({2, 3, 3}, ge::DT_STRING));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringNormalizer, string_normalizer_verify_failed3){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({2, 3}, ge::DT_STRING));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringNormalizer, string_normalizer_verify_failed4){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({1, 3}, ge::DT_STRING));
  op.SetAttr("case_change_action", "ERROR");
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringNormalizer, string_normalizer_verify_success){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({1, 3}, ge::DT_STRING));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringNormalizer, string_normalizer__infershape){
  ge::op::StringNormalizer op;
  op.UpdateInputDesc("input", create_desc({1,3}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_STRING);
  std::vector<int64_t> expected_output_shape = {1, 3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}