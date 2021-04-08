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

class StringToHashBucket : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringToHashBucket SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringToHashBucket TearDown" << std::endl;
  }
};

TEST_F(StringToHashBucket, string_to_hash_bucket_infer_shape) {
  ge::op::StringToHashBucket op;
  op.UpdateInputDesc("string_tensor", create_desc({2}, ge::DT_STRING));
  op.SetAttr("num_buckets", 1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringToHashBucket, string_to_hash_bucket_fast_infer_shape) {
  ge::op::StringToHashBucketFast op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  op.SetAttr("num_buckets", 1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringToHashBucket, string_to_hash_bucket_strong_infer_shape) {
  ge::op::StringToHashBucketStrong op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  op.SetAttr("num_buckets", 1);
  op.SetAttr("key", {1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringToHashBucket, string_to_hash_bucket_strong_infer_shape_error1) {
  ge::op::StringToHashBucketStrong op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_INT32));
  op.SetAttr("num_buckets", 1);
  op.SetAttr("key", {1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

