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

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

class AvgPoolV2Grad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolV2Grad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolV2Grad_UT TearDown" << std::endl;
  }
};

TEST_F(AvgPoolV2Grad_UT, VerifyAvgPoolV2Grad_001) {
  ge::op::AvgPoolV2Grad op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("orig_input_shape", tensor_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2Grad_UT, InfershapeAvgPoolV2Grad_001) {
  ge::op::AvgPoolV2Grad op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("orig_input_shape", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2Grad_UT, InfershapeAvgPoolV2Grad_002) {
  ge::op::AvgPoolV2Grad op;
  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_orig_input_shape(const_op);
  op.UpdateInputDesc("orig_input_shape", const_desc);
  op.UpdateInputDesc("input_grad", create_desc({2,2},ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {1, 1, 1, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}