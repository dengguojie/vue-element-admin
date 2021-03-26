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

class StringOp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringOp SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringOp TearDown" << std::endl;
  }
};

TEST_F(StringOp, string_n_grams_infer_shape) {
  ge::op::StringNGrams op;
  op.UpdateInputDesc("data", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("data_splits", create_desc({2}, ge::DT_INT32));
  op.SetAttr("separator", "");
  op.SetAttr("ngram_widths", "{0}");
  op.SetAttr("left_pad", "");
  op.SetAttr("right_pad", "");
  op.SetAttr("pad_width", 0);
  op.SetAttr("preserve_short_sequences", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, string_n_grams_infer_shape_error1) {
  ge::op::StringNGrams op;
  op.UpdateInputDesc("data", create_desc({2,1}, ge::DT_STRING));
  op.UpdateInputDesc("data_splits", create_desc({2}, ge::DT_INT32));
  op.SetAttr("separator", "");
  op.SetAttr("ngram_widths", "{0}");
  op.SetAttr("left_pad", "");
  op.SetAttr("right_pad", "");
  op.SetAttr("pad_width", 0);
  op.SetAttr("preserve_short_sequences", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringOp, string_n_grams_infer_shape_error2) {
  ge::op::StringNGrams op;
  op.UpdateInputDesc("data", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("data_splits", create_desc({2,1}, ge::DT_INT32));
  op.SetAttr("separator", "");
  op.SetAttr("ngram_widths", "{0}");
  op.SetAttr("left_pad", "");
  op.SetAttr("right_pad", "");
  op.SetAttr("pad_width", 0);
  op.SetAttr("preserve_short_sequences", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringOp, unicode_decode_with_offsets_infer_shape) {
  ge::op::UnicodeDecodeWithOffsets op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.SetAttr("input_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  op.SetAttr("replace_control_characters", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, unicode_decode_infer_shape) {
  ge::op::UnicodeDecode op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.SetAttr("input_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  op.SetAttr("replace_control_characters", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, unicode_transcode_infer_shape) {
  ge::op::UnicodeTranscode op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_STRING));
  op.SetAttr("input_encoding", "UTF-8");
  op.SetAttr("output_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  op.SetAttr("replace_control_characters", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, unicode_encode_infer_shape) {
  ge::op::UnicodeEncode op;
  op.UpdateInputDesc("input_values", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("input_splits", create_desc({2}, ge::DT_INT32));
  op.SetAttr("output_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, unicode_encode_infer_shape_error1) {
  ge::op::UnicodeEncode op;
  op.UpdateInputDesc("input_values", create_desc({2,1}, ge::DT_INT32));
  op.UpdateInputDesc("input_splits", create_desc({2}, ge::DT_INT32));
  op.SetAttr("output_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StringOp, unicode_encode_infer_shape_error2) {
  ge::op::UnicodeEncode op;
  op.UpdateInputDesc("input_values", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("input_splits", create_desc({2,1}, ge::DT_INT32));
  op.SetAttr("output_encoding", "UTF-8");
  op.SetAttr("errors", "replace");
  op.SetAttr("replacement_char", 65533);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}