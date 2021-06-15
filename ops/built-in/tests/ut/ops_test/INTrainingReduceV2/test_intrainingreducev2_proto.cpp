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

class in_training_reduce_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "in_training_reduce_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_training_reduce_v2 TearDown" << std::endl;
  }
};

TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_1) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc(
      "x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {4, 64, 64, 16}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);
}
