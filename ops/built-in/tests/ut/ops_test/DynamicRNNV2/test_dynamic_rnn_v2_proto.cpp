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

class DynamicRnnV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_rnn_v2 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_v2 test TearDown" << std::endl;
  }
};

TEST_F(DynamicRnnV2Test, dynamic_rnn_test_case_1) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicRNNV2 rnn_op;
  ge::TensorDesc XDesc;
  ge::Shape xShape({t, batch, inputSize});
  XDesc.SetDataType(ge::DT_FLOAT16);
  XDesc.SetShape(xShape);
  XDesc.SetOriginShape(xShape);

  ge::TensorDesc WiDesc;
  ge::TensorDesc WhDesc;
  ge::Shape wiShape({inputSize, 4 * outputSize});
  ge::Shape whShape({outputSize, 4 * outputSize});
  WiDesc.SetDataType(ge::DT_FLOAT16);
  WhDesc.SetDataType(ge::DT_FLOAT16);
  WiDesc.SetShape(wiShape);
  WhDesc.SetShape(whShape);
  WiDesc.SetOriginShape(wiShape);
  WhDesc.SetOriginShape(whShape);

  ge::TensorDesc BDesc;
  ge::Shape bShape({4 * outputSize});
  BDesc.SetDataType(ge::DT_FLOAT16);
  BDesc.SetShape(bShape);
  BDesc.SetOriginShape(bShape);

  rnn_op.UpdateInputDesc("x", XDesc);
  rnn_op.UpdateInputDesc("weight_input", WiDesc);
  rnn_op.UpdateInputDesc("weight_hidden", WhDesc);
  rnn_op.UpdateInputDesc("b", BDesc);

  auto status = rnn_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = rnn_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = rnn_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {t, batch, outputSize};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicRnnV2Test, dynamic_rnn_test_case_2) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicRNNV2 rnn_op;
  ge::TensorDesc XDesc;
  ge::Shape xShape({t, batch, inputSize, 1});
  XDesc.SetDataType(ge::DT_FLOAT16);
  XDesc.SetShape(xShape);
  XDesc.SetOriginShape(xShape);

  ge::TensorDesc WiDesc;
  ge::TensorDesc WhDesc;
  ge::Shape wiShape({inputSize, 4 * outputSize});
  ge::Shape whShape({outputSize, 4 * outputSize});
  WiDesc.SetDataType(ge::DT_FLOAT16);
  WhDesc.SetDataType(ge::DT_FLOAT16);
  WiDesc.SetShape(wiShape);
  WhDesc.SetShape(whShape);
  WiDesc.SetOriginShape(wiShape);
  WhDesc.SetOriginShape(whShape);

  ge::TensorDesc BDesc;
  ge::Shape bShape({4 * outputSize});
  BDesc.SetDataType(ge::DT_FLOAT16);
  BDesc.SetShape(bShape);
  BDesc.SetOriginShape(bShape);

  rnn_op.UpdateInputDesc("x", XDesc);
  rnn_op.UpdateInputDesc("weight_input", WiDesc);
  rnn_op.UpdateInputDesc("weight_hidden", WhDesc);
  rnn_op.UpdateInputDesc("b", BDesc);

  auto ret = rnn_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}