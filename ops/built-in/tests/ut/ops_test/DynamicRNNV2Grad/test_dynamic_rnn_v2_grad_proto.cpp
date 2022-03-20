/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicRnnV2GradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_rnn_v2_grad test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_v2_grad test TearDown" << std::endl;
  }
};

TEST_F(DynamicRnnV2GradTest, dynamic_rnn_v2_grad_normal_shape) {
  int batch = 16;
  int inputSize = 32;
  int hiddenSize = 48;
  ge::op::DynamicRNNV2Grad rnn_op;
  ge::TensorDesc xDesc;
  ge::Shape xShape({batch, inputSize});
  xDesc.SetDataType(ge::DT_FLOAT16);
  xDesc.SetShape(xShape);
  xDesc.SetOriginShape(xShape);

  ge::TensorDesc wiDesc;
  ge::TensorDesc whDesc;
  ge::Shape wiShape({inputSize, 4 * hiddenSize});
  ge::Shape whShape({hiddenSize, 4 * hiddenSize});
  wiDesc.SetDataType(ge::DT_FLOAT16);
  whDesc.SetDataType(ge::DT_FLOAT16);
  wiDesc.SetShape(wiShape);
  whDesc.SetShape(whShape);
  wiDesc.SetOriginShape(wiShape);
  whDesc.SetOriginShape(whShape);

  ge::TensorDesc yDesc;
  ge::Shape yShape({batch, hiddenSize});
  yDesc.SetDataType(ge::DT_FLOAT16);
  yDesc.SetShape(yShape);
  yDesc.SetOriginShape(yShape);

  ge::TensorDesc cellDesc;
  ge::Shape cellShape({batch, hiddenSize});
  cellDesc.SetDataType(ge::DT_FLOAT16);
  cellDesc.SetShape(cellShape);
  cellDesc.SetOriginShape(cellShape);

  rnn_op.UpdateInputDesc("x", xDesc);
  rnn_op.UpdateInputDesc("w_x", wiDesc);
  rnn_op.UpdateInputDesc("w_h", whDesc);
  rnn_op.UpdateInputDesc("y", yDesc);
  rnn_op.UpdateInputDesc("init_h", cellDesc);
  rnn_op.UpdateInputDesc("init_c", cellDesc);
  rnn_op.UpdateInputDesc("h", yDesc);
  rnn_op.UpdateInputDesc("c", yDesc);
  rnn_op.UpdateInputDesc("dy", yDesc);
  rnn_op.UpdateInputDesc("dh", cellDesc);
  rnn_op.UpdateInputDesc("dc", cellDesc);
  rnn_op.UpdateInputDesc("i", yDesc);
  rnn_op.UpdateInputDesc("j", yDesc);
  rnn_op.UpdateInputDesc("f", yDesc);
  rnn_op.UpdateInputDesc("o", yDesc);
  rnn_op.UpdateInputDesc("tanhct", yDesc);

  auto status = rnn_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = rnn_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = rnn_op.GetOutputDesc("dx");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {batch, inputSize};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicRnnV2GradTest, dynamic_rnn_v2_grad_invalid_time_step) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int hiddenSize = 48;
  ge::op::DynamicRNNV2Grad rnn_op;
  ge::TensorDesc xDesc;
  ge::Shape xShape({t, batch, inputSize});
  xDesc.SetDataType(ge::DT_FLOAT16);
  xDesc.SetShape(xShape);
  xDesc.SetOriginShape(xShape);

  ge::TensorDesc wiDesc;
  ge::TensorDesc whDesc;
  ge::Shape wiShape({inputSize, 4 * hiddenSize});
  ge::Shape whShape({hiddenSize, 4 * hiddenSize});
  wiDesc.SetDataType(ge::DT_FLOAT16);
  whDesc.SetDataType(ge::DT_FLOAT16);
  wiDesc.SetShape(wiShape);
  whDesc.SetShape(whShape);
  wiDesc.SetOriginShape(wiShape);
  whDesc.SetOriginShape(whShape);

  ge::TensorDesc yDesc;
  ge::Shape yShape({t, batch, hiddenSize});
  yDesc.SetDataType(ge::DT_FLOAT16);
  yDesc.SetShape(yShape);
  yDesc.SetOriginShape(yShape);

  ge::TensorDesc cellDesc;
  ge::Shape cellShape({1, batch, hiddenSize});
  cellDesc.SetDataType(ge::DT_FLOAT16);
  cellDesc.SetShape(cellShape);
  cellDesc.SetOriginShape(cellShape);

  rnn_op.UpdateInputDesc("x", xDesc);
  rnn_op.UpdateInputDesc("w_x", wiDesc);
  rnn_op.UpdateInputDesc("w_h", whDesc);
  rnn_op.UpdateInputDesc("y", yDesc);
  rnn_op.UpdateInputDesc("init_h", cellDesc);
  rnn_op.UpdateInputDesc("init_c", cellDesc);
  rnn_op.UpdateInputDesc("h", yDesc);
  rnn_op.UpdateInputDesc("c", yDesc);
  rnn_op.UpdateInputDesc("dy", yDesc);
  rnn_op.UpdateInputDesc("dh", cellDesc);
  rnn_op.UpdateInputDesc("dc", cellDesc);
  rnn_op.UpdateInputDesc("i", yDesc);
  rnn_op.UpdateInputDesc("j", yDesc);
  rnn_op.UpdateInputDesc("f", yDesc);
  rnn_op.UpdateInputDesc("o", yDesc);
  rnn_op.UpdateInputDesc("tanhct", yDesc);

  auto ret = rnn_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV2GradTest, dynamic_rnn_v2_grad_invalid_x) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int hiddenSize = 48;
  ge::op::DynamicRNNV2Grad rnn_op;
  ge::TensorDesc xDesc;
  ge::Shape xShape({t, batch, inputSize, 1});
  xDesc.SetDataType(ge::DT_FLOAT16);
  xDesc.SetShape(xShape);
  xDesc.SetOriginShape(xShape);

  ge::TensorDesc wiDesc;
  ge::TensorDesc whDesc;
  ge::Shape wiShape({inputSize, 4 * hiddenSize});
  ge::Shape whShape({hiddenSize, 4 * hiddenSize});
  wiDesc.SetDataType(ge::DT_FLOAT16);
  whDesc.SetDataType(ge::DT_FLOAT16);
  wiDesc.SetShape(wiShape);
  whDesc.SetShape(whShape);
  wiDesc.SetOriginShape(wiShape);
  whDesc.SetOriginShape(whShape);

  ge::TensorDesc yDesc;
  ge::Shape yShape({t, batch, hiddenSize});
  yDesc.SetDataType(ge::DT_FLOAT16);
  yDesc.SetShape(yShape);
  yDesc.SetOriginShape(yShape);

  ge::TensorDesc cellDesc;
  ge::Shape cellShape({1, batch, hiddenSize});
  cellDesc.SetDataType(ge::DT_FLOAT16);
  cellDesc.SetShape(cellShape);
  cellDesc.SetOriginShape(cellShape);

  rnn_op.UpdateInputDesc("x", xDesc);
  rnn_op.UpdateInputDesc("w_x", wiDesc);
  rnn_op.UpdateInputDesc("w_h", whDesc);
  rnn_op.UpdateInputDesc("y", yDesc);
  rnn_op.UpdateInputDesc("init_h", cellDesc);
  rnn_op.UpdateInputDesc("init_c", cellDesc);
  rnn_op.UpdateInputDesc("h", yDesc);
  rnn_op.UpdateInputDesc("c", yDesc);
  rnn_op.UpdateInputDesc("dy", yDesc);
  rnn_op.UpdateInputDesc("dh", cellDesc);
  rnn_op.UpdateInputDesc("dc", cellDesc);
  rnn_op.UpdateInputDesc("i", yDesc);
  rnn_op.UpdateInputDesc("j", yDesc);
  rnn_op.UpdateInputDesc("f", yDesc);
  rnn_op.UpdateInputDesc("o", yDesc);
  rnn_op.UpdateInputDesc("tanhct", yDesc);

  auto ret = rnn_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
