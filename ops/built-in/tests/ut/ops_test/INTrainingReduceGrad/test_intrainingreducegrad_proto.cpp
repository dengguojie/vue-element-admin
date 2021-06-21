/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "op_proto_test_util.h"

class in_training_reduce_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "in_training_reduce_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_training_reduce_grad TearDown" << std::endl;
  }
};

TEST_F(in_training_reduce_grad, in_training_reduce_grad_infershape_diff_test_1) {
  ge::op::INTrainingReduceGrad op;

  op.UpdateInputDesc("dy", create_desc_with_ori({2, 128, 128, 128, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                {2, 128, 128, 128, 32}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("x", create_desc_with_ori({2, 128, 128, 128, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                               {2, 128, 128, 128, 32}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("variance", create_desc_with_ori({2, 1, 1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                      {2, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("mean", create_desc_with_ori({2, 1, 1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC, {2, 1, 1, 1, 32},
                                                  ge::FORMAT_NDHWC));
  op.UpdateInputDesc("res_gamma", create_desc_with_ori({2, 1, 1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                       {2, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("res_beta", create_desc_with_ori({2, 1, 1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                      {2, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("gamma", create_desc_with_ori({1, 1, 1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC, {1, 1, 1, 1, 32},
                                                   ge::FORMAT_NDHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("pd_x");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {2, 128, 128, 128, 32};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);
}
