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

class FractionalMaxPoolGrad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FractionalMaxPoolGrad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FractionalMaxPoolGrad_UT TearDown" << std::endl;
  }
};

TEST_F(FractionalMaxPoolGrad_UT, InfershapeFractionalMaxPoolGrad_001) {
  ge::op::FractionalMaxPoolGrad op;
  op.UpdateInputDesc("orig_input", create_desc({2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FractionalMaxPoolGrad_UT, InfershapeFractionalMaxPoolGrad_002) {
  ge::op::FractionalMaxPoolGrad op;
  op.UpdateInputDesc("orig_input", create_desc({2, 2, 2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_y = {2, 2, 2, 2};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}