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

TEST_F(StringOp, string_strip_infer_shape) {
  ge::op::StringStrip op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, string_length_infer_shape) {
  ge::op::StringLength op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  op.SetAttr("unit", "BYTE");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, regex_full_match_infer_shape) {
  ge::op::RegexFullMatch op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("pattern", create_desc({}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, as_string_infer_shape) {
  ge::op::AsString op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));
  op.SetAttr("precision", -1);
  op.SetAttr("scientific", false);
  op.SetAttr("shortest", false);
  op.SetAttr("width", -1);
  op.SetAttr("fill", "");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, decode_base64_infer_shape) {
  ge::op::DecodeBase64 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_STRING));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(StringOp, string_join_infer_shape) {
  ge::op::StringJoin op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({2, 100, 4},
                                             ge::DT_STRING, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("separator", "");
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