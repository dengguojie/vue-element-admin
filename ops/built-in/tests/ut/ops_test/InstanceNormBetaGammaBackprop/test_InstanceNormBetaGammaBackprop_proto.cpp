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
#include "nn_norm_ops.h"
#include "op_proto_test_util.h"

class beta_gamma_backprop : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "beta_gamma_backprop SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "beta_gamma_backprop TearDown" << std::endl;
  }
};

TEST_F(beta_gamma_backprop, beta_gamma_backprop_infershape_diff_test_1) {
  ge::op::InstanceNormBetaGammaBackprop op;

  op.UpdateInputDesc("dy", create_desc_with_ori({4, 64, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
                                                {4, 64, 64, 64, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("res_for_gamma", create_desc_with_ori({4, 64, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                                           {4, 64, 64, 64, 16}, ge::FORMAT_NDHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("pd_gamma");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("pd_beta");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_batch_mean_output_shape = {16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);
}
