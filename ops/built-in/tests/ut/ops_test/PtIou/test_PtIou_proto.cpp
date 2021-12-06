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
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PtIouTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PtIouTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PtIouTest_UT TearDown" << std::endl;
  }
};

TEST_F(PtIouTest_UT, InferShapePtIou_001) {
  ge::op::PtIou op;
  op.UpdateInputDesc("bboxes", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("gtboxes", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto lap_output_desc = op.GetOutputDesc("overlap");
  std::vector<int64_t> expected_output_shape = {4, 4};
  EXPECT_EQ(lap_output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(lap_output_desc.GetDataType(), ge::DT_FLOAT16);
}