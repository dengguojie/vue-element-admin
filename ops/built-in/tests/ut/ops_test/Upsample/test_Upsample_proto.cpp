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
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"

class Upsample_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Upsample_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Upsample_UT TearDown" << std::endl;
  }
};

TEST_F(Upsample_UT, InfershapeUpsample_001) {
  ge::op::Upsample op;
  auto tensor_desc = create_desc({2, 2, 3, 4}, ge::DT_FLOAT);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("stride_h", {});
  op.SetAttr("stride_w", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {2, 2, 6, 8};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}