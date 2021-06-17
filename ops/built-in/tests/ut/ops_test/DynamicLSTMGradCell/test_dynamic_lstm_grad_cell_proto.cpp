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

class DynamicLSTMGradCellTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_lstm_grad_cell test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_lstm_grad_cell test TearDown" << std::endl;
  }
};

TEST_F(DynamicLSTMGradCellTest, dynamic_lstm_v2_test_case_1_success) {
  int t = 2;
  int batch = 4;
  int input_size = 1;
  int hiddenSize = 1;
  ge::op::DynamicLSTMGradCell dynamic_lstm_v2_op;

  ge::TensorDesc init_c_desc;
  ge::Shape init_c_shape({1, input_size, batch, 16, 16});
  init_c_desc.SetDataType(ge::DT_FLOAT16);
  init_c_desc.SetShape(init_c_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("init_c", init_c_desc);

  ge::TensorDesc c_desc;
  ge::Shape c_shape({t, input_size, batch, 16, 16});
  c_desc.SetDataType(ge::DT_FLOAT16);
  c_desc.SetShape(c_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("c", c_desc);

  ge::TensorDesc dy_desc;
  ge::Shape dy_shape({t, input_size, batch, 16, 16});
  dy_desc.SetDataType(ge::DT_FLOAT16);
  dy_desc.SetShape(dy_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("dy", dy_desc);

  ge::TensorDesc dh_desc;
  ge::Shape dh_shape({1, input_size, batch, 16, 16});
  dh_desc.SetDataType(ge::DT_FLOAT16);
  dh_desc.SetShape(dh_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("dh", dh_desc);

  ge::TensorDesc dc_desc;
  ge::Shape dc_shape({1, input_size, batch, 16, 16});
  dc_desc.SetDataType(ge::DT_FLOAT16);
  dc_desc.SetShape(dc_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("dc", dc_desc);

  ge::TensorDesc i_desc;
  ge::Shape i_shape({t, input_size, batch, 16, 16});
  i_desc.SetDataType(ge::DT_FLOAT16);
  i_desc.SetShape(i_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("i", i_desc);

  ge::TensorDesc j_desc;
  ge::Shape j_shape({t, input_size, batch, 16, 16});
  j_desc.SetDataType(ge::DT_FLOAT16);
  j_desc.SetShape(j_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("j", j_desc);

  ge::TensorDesc f_desc;
  ge::Shape f_shape({t, input_size, batch, 16, 16});
  f_desc.SetDataType(ge::DT_FLOAT16);
  f_desc.SetShape(f_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("f", f_desc);

  ge::TensorDesc o_desc;
  ge::Shape o_shape({t, input_size, batch, 16, 16});
  o_desc.SetDataType(ge::DT_FLOAT16);
  o_desc.SetShape(o_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("o", o_desc);

  ge::TensorDesc tanhct_desc;
  ge::Shape tanhct_shape({t, input_size, batch, 16, 16});
  tanhct_desc.SetDataType(ge::DT_FLOAT16);
  tanhct_desc.SetShape(tanhct_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("tanhct", tanhct_desc);

  ge::TensorDesc mask_desc;
  ge::Shape mask_shape({t, input_size, batch, 16, 16});
  mask_desc.SetDataType(ge::DT_FLOAT16);
  mask_desc.SetShape(mask_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("mask", mask_desc);

  ge::TensorDesc t_state_desc;
  ge::Shape t_state_shape({1});
  t_state_desc.SetDataType(ge::DT_FLOAT16);
  t_state_desc.SetShape(t_state_shape);
  dynamic_lstm_v2_op.UpdateInputDesc("t_state", t_state_desc);

  dynamic_lstm_v2_op.SetAttr("forget_bias", (float)1.0);
  dynamic_lstm_v2_op.SetAttr("activation", "tanh");
  dynamic_lstm_v2_op.SetAttr("direction", "UNIDIRECTIONAL");
  dynamic_lstm_v2_op.SetAttr("gate_order", "ijfo");

  auto ret = dynamic_lstm_v2_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = dynamic_lstm_v2_op.GetOutputDesc("dgate");
  EXPECT_EQ(output_desc.GetDataType(), 1);
  std::vector<int64_t> expected_output_shape = {1, 4 * input_size, batch, 16, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
