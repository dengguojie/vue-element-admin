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
 * @file test_decode_jpeg_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "parsing_ops.h"

class DecodeCSV : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeCSV SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeCSV TearDown" << std::endl;
  }
};

TEST_F(DecodeCSV, decode_csv_infer_shape_success) {
  ge::op::DecodeCSV op;

  op.UpdateInputDesc("records", create_desc({1, 2, 3, 4}, ge::DT_STRING));

  std::vector<ge::DataType> OUT_TYPE {ge::DT_STRING, ge::DT_STRING, ge::DT_STRING, ge::DT_STRING};
  op.SetAttr("OUT_TYPE", OUT_TYPE);

  const int32_t record_defaults_size = 4;
  op.create_dynamic_input_record_defaults(record_defaults_size);
  for (int i = 0; i < record_defaults_size; ++i) {
    op.UpdateDynamicInputDesc("record_defaults", i, create_desc({-1}, ge::DT_FLOAT));
  }

  const int32_t output_size = 4;
  op.create_dynamic_output_output(output_size);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DecodeCSV, decode_csv_infer_shape_failed) {
  ge::op::DecodeCSV op;

  op.UpdateInputDesc("records", create_desc({1, 2}, ge::DT_STRING));

  std::vector<ge::DataType> OUT_TYPE {ge::DT_STRING, ge::DT_STRING, ge::DT_STRING, ge::DT_STRING};
  op.SetAttr("OUT_TYPE", OUT_TYPE);

  op.create_dynamic_input_record_defaults(2);
  op.UpdateDynamicInputDesc("record_defaults", 0, create_desc({-1, -1}, ge::DT_FLOAT));
  op.UpdateDynamicInputDesc("record_defaults", 1, create_desc({-1, -1}, ge::DT_FLOAT));

  op.create_dynamic_output_output(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}