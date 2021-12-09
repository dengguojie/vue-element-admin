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
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"
#include "graph/compute_graph.h"
#include "array_ops.h"

class FractionalAvgPoolGrad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FractionalAvgPoolGrad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FractionalAvgPoolGrad_UT TearDown" << std::endl;
  }
};

TEST_F(FractionalAvgPoolGrad_UT, InfershapeFractionalAvgPoolGrad_001) {
  ge::op::FractionalAvgPoolGrad op;
  op.UpdateInputDesc("orig_input_tensor_shape", create_desc({2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FractionalAvgPoolGrad_UT, InfershapeFractionalAvgPoolGrad_002) {
  ge::op::FractionalAvgPoolGrad op;
  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_orig_input_tensor_shape(const_op);
  op.UpdateInputDesc("orig_input_tensor_shape", const_desc);
  op.UpdateInputDesc("out_backprop", create_desc({2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_y = {1, 1, 1, 1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}