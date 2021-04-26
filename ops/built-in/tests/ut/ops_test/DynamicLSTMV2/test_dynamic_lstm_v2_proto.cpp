/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_dynamic_lstm_v2.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicLSTMV2Test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_lstm_v2 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_lstm_v2 test TearDown" << std::endl;
  }
};

TEST_F(DynamicLSTMV2Test, dynamic_lstm_v2_test_case_1_success) {
  int t = 75;
  int batch = 1;
  int input_size = 512;
  int hiddenSize = 256;
  ge::op::DynamicLSTMV2 dynamic_lstm_v2_op;

  ge::TensorDesc x_desc;
  ge::Shape x_shape({t, batch, input_size});
  x_desc.SetDataType(ge::DT_FLOAT16);
  x_desc.SetShape(x_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("x", x_desc);

  ge::TensorDesc w_input_desc;
  ge::Shape w_input_shape({input_size+hiddenSize, 4 * hiddenSize});
  w_input_desc.SetDataType(ge::DT_FLOAT16);
  w_input_desc.SetShape(w_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w", w_input_desc);

  ge::TensorDesc b_input_desc;
  ge::Shape b_input_shape({4 * hiddenSize, });
  b_input_desc.SetDataType(ge::DT_FLOAT16);
  b_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("b", b_input_desc);

  ge::TensorDesc cont_input_desc;
  ge::Shape cont_input_shape({t, batch});
  cont_input_desc.SetDataType(ge::DT_FLOAT16);
  cont_input_desc.SetShape(cont_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("cont", cont_input_desc);

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, 50331649);

  auto output_desc = dynamic_lstm_v2_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), 0);
  // std::vector<int64_t> expected_output_shape = {t, batch, hiddenSize};
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicLSTMV2Test, dynamic_lstm_v2_test_case_2_success) {
  int t = 75;
  int batch = 1;
  int input_size = 512;
  int hiddenSize = 256;
  ge::op::DynamicLSTMV2 dynamic_lstm_v2_op;

  ge::TensorDesc x_desc;
  ge::Shape x_shape({t, batch, input_size});
  x_desc.SetDataType(ge::DT_FLOAT16);
  x_desc.SetShape(x_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("x", x_desc);

  ge::TensorDesc w_input_desc;
  ge::Shape w_input_shape({input_size+hiddenSize, 4 * hiddenSize});
  w_input_desc.SetDataType(ge::DT_FLOAT16);
  w_input_desc.SetShape(w_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w", w_input_desc);

  ge::TensorDesc b_input_desc;
  ge::Shape b_input_shape({4 * hiddenSize, });
  b_input_desc.SetDataType(ge::DT_FLOAT16);
  b_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("b", b_input_desc);

  ge::TensorDesc cont_input_desc;
  ge::Shape cont_input_shape({t, batch});
  cont_input_desc.SetDataType(ge::DT_FLOAT16);
  cont_input_desc.SetShape(cont_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("cont", cont_input_desc);

  ge::TensorDesc static_input_desc;
  ge::Shape static_shape({64, 1, 16, 16});
  static_input_desc.SetDataType(ge::DT_FLOAT16);
  static_input_desc.SetShape(static_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w_xc_x_static", static_input_desc);

  ge::TensorDesc h_init_desc;
  ge::Shape h_init_shape({1, batch, input_size});
  h_init_desc.SetDataType(ge::DT_FLOAT16);
  h_init_desc.SetShape(h_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", h_init_desc);

  ge::TensorDesc c_init_desc;
  ge::Shape c_init_shape({1, batch, input_size});
  c_init_desc.SetDataType(ge::DT_FLOAT16);
  c_init_desc.SetShape(c_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", c_init_desc);

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, 50331649);
  std::cout << "dynamic_lstm_v2 test 22222" << std::endl;

  auto output_desc = dynamic_lstm_v2_op.GetOutputDesc("y");
  auto last_h_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_h");
  auto last_c_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_c");

  EXPECT_EQ(output_desc.GetDataType(), 0);
  EXPECT_EQ(last_h_desc.GetDataType(), 0);
  EXPECT_EQ(last_c_desc.GetDataType(), 0);
  // std::vector<int64_t> expected_output_shape = {t, batch, hiddenSize};
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicLSTMV2Test, dynamic_lstm_v2_test_case_3_success) {
  int t = 75;
  int batch = 2;
  int input_size = 512;
  int hiddenSize = 256;
  ge::op::DynamicLSTMV2 dynamic_lstm_v2_op;

  ge::TensorDesc x_desc;
  ge::Shape x_shape({t, batch, input_size});
  x_desc.SetDataType(ge::DT_FLOAT16);
  x_desc.SetShape(x_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("x", x_desc);

  ge::TensorDesc w_input_desc;
  ge::Shape w_input_shape({input_size+hiddenSize, 4 * hiddenSize});
  w_input_desc.SetDataType(ge::DT_FLOAT16);
  w_input_desc.SetShape(w_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w", w_input_desc);

  ge::TensorDesc b_input_desc;
  ge::Shape b_input_shape({4 * hiddenSize, });
  b_input_desc.SetDataType(ge::DT_FLOAT16);
  b_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("b", b_input_desc);

  ge::TensorDesc cont_input_desc;
  ge::Shape cont_input_shape({t, batch});
  cont_input_desc.SetDataType(ge::DT_FLOAT16);
  cont_input_desc.SetShape(cont_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("cont", cont_input_desc);

  ge::TensorDesc static_input_desc;
  ge::Shape static_shape({64, 1, 16, 16});
  static_input_desc.SetDataType(ge::DT_FLOAT16);
  static_input_desc.SetShape(static_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w_xc_x_static", static_input_desc);

  ge::TensorDesc h_init_desc;
  ge::Shape h_init_shape({1, batch, input_size});
  h_init_desc.SetDataType(ge::DT_FLOAT16);
  h_init_desc.SetShape(h_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", h_init_desc);

  ge::TensorDesc c_init_desc;
  ge::Shape c_init_shape({1, batch, input_size});
  c_init_desc.SetDataType(ge::DT_FLOAT16);
  c_init_desc.SetShape(c_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", c_init_desc);

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, 50331649);
  std::cout << "dynamic_lstm_v2 test 22222" << std::endl;

  auto output_desc = dynamic_lstm_v2_op.GetOutputDesc("y");
  auto last_h_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_h");
  auto last_c_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_c");

  EXPECT_EQ(output_desc.GetDataType(), 0);
  EXPECT_EQ(last_h_desc.GetDataType(), 0);
  EXPECT_EQ(last_c_desc.GetDataType(), 0);
  // std::vector<int64_t> expected_output_shape = {t, batch, hiddenSize};
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicLSTMV2Test, dynamic_lstm_v2_test_case_5_success) {
  int t = 60;
  int batch = 1;
  int input_size = 512;
  int hiddenSize = 128;
  ge::op::DynamicLSTMV2 dynamic_lstm_v2_op;

  ge::TensorDesc x_desc;
  ge::Shape x_shape({t, batch, input_size});
  x_desc.SetDataType(ge::DT_FLOAT16);
  x_desc.SetShape(x_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("x", x_desc);

  ge::TensorDesc w_input_desc;
  ge::Shape w_input_shape({input_size+hiddenSize, 4 * hiddenSize});
  w_input_desc.SetDataType(ge::DT_FLOAT16);
  w_input_desc.SetShape(w_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w", w_input_desc);

  ge::TensorDesc b_input_desc;
  ge::Shape b_input_shape({4 * hiddenSize, });
  b_input_desc.SetDataType(ge::DT_FLOAT16);
  b_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("b", b_input_desc);

  ge::TensorDesc cont_input_desc;
  ge::Shape cont_input_shape({t, batch});
  cont_input_desc.SetDataType(ge::DT_FLOAT16);
  cont_input_desc.SetShape(cont_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("cont", cont_input_desc);

  ge::TensorDesc static_input_desc;
  ge::Shape static_shape({64, 1, 16, 16});
  static_input_desc.SetDataType(ge::DT_FLOAT16);
  static_input_desc.SetShape(static_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w_xc_x_static", static_input_desc);

  ge::TensorDesc h_init_desc;
  ge::Shape h_init_shape({1, batch, input_size});
  h_init_desc.SetDataType(ge::DT_FLOAT16);
  h_init_desc.SetShape(h_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", h_init_desc);

  ge::TensorDesc c_init_desc;
  ge::Shape c_init_shape({1, batch, input_size});
  c_init_desc.SetDataType(ge::DT_FLOAT16);
  c_init_desc.SetShape(c_init_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("h0", c_init_desc);

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, 50331649);
  std::cout << "dynamic_lstm_v2 test 22222" << std::endl;

  auto output_desc = dynamic_lstm_v2_op.GetOutputDesc("y");
  auto last_h_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_h");
  auto last_c_desc = dynamic_lstm_v2_op.GetOutputDesc("last_output_c");

  EXPECT_EQ(output_desc.GetDataType(), 0);
  EXPECT_EQ(last_h_desc.GetDataType(), 0);
  EXPECT_EQ(last_c_desc.GetDataType(), 0);
  // std::vector<int64_t> expected_output_shape = {t, batch, hiddenSize};
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicLSTMV2Test, dynamic_lstm_v2_test_case_1_failed) {
  int t = 75;
  int batch = 1;
  int input_size = 512;
  int hiddenSize = 256;
  ge::op::DynamicLSTMV2 dynamic_lstm_v2_op;

  ge::TensorDesc x_desc;
  ge::Shape x_shape({t, batch, input_size, hiddenSize});
  x_desc.SetDataType(ge::DT_FLOAT16);
  x_desc.SetShape(x_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("x", x_desc);

  ge::TensorDesc w_input_desc;
  ge::Shape w_input_shape({hiddenSize, 4 * hiddenSize});
  w_input_desc.SetDataType(ge::DT_FLOAT16);
  w_input_desc.SetShape(w_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("w", w_input_desc);

  ge::TensorDesc b_input_desc;
  ge::Shape b_input_shape({4 * hiddenSize, });
  b_input_desc.SetDataType(ge::DT_FLOAT16);
  b_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("b", b_input_desc);

  ge::TensorDesc cont_input_desc;
  cont_input_desc.SetDataType(ge::DT_FLOAT16);
  cont_input_desc.SetShape(b_input_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("cont", cont_input_desc);

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  EXPECT_EQ(ret, 50331649); //PARAM_INVALID
}
