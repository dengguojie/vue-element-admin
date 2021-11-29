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
#include "elewise_calculation_ops.h"

class ConfusionMulGrad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConfusionMulGrad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConfusionMulGrad_UT TearDown" << std::endl;
  }
};

TEST_F(ConfusionMulGrad_UT, InfershapeConfusionMulGrad_test_01) {
  ge::op::ConfusionMulGrad op;
  op.UpdateInputDesc("input0", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input1", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ConfusionMulGrad_UT, InfershapeConfusionMulGrad_test_02) {
  ge::op::ConfusionMulGrad op;
  op.UpdateInputDesc("input0", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input1", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.SetAttr("axes", 3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ConfusionMulGrad_UT, InfershapeConfusionMulGrad_test_03) {
  ge::op::ConfusionMulGrad op;
  op.UpdateInputDesc("input0", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input1", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.SetAttr("keep_dims", "False");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}