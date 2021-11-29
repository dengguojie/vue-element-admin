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

#include <gtest/gtest.h>              // NOLINT
#include <iostream>                   // NOLINT
#include "op_proto_test_util.h"       // NOLINT
#include "elewise_calculation_ops.h"  // NOLINT

class FakeQuantWithMinMaxVarsGradient_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FakeQuantWithMinMaxVarsGradient_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FakeQuantWithMinMaxVarsGradient_UT TearDown" << std::endl;
  }
};

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_001) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_x = op.GetOutputDesc("backprops_wrt_x");
  EXPECT_EQ(output_desc_x.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_x = {4, 3, 1};
  EXPECT_EQ(output_desc_x.GetShape().GetDims(), expected_output_shape_x);

  auto output_desc_min = op.GetOutputDesc("backprops_wrt_min");
  EXPECT_EQ(output_desc_min.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_min = {1};
  EXPECT_EQ(output_desc_min.GetShape().GetDims(), expected_output_shape_min);

  auto output_desc_max = op.GetOutputDesc("backprops_wrt_max");
  EXPECT_EQ(output_desc_max.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_max = {1};
  EXPECT_EQ(output_desc_max.GetShape().GetDims(), expected_output_shape_max);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_002) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  std::string num_bits = "zero";
  op.SetAttr("num_bits", num_bits);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_003) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  int8_t narrow_range = 8;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_004) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_005) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_006) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_007) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int64_t num_bits = 30;
  op.SetAttr("num_bits", num_bits);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_008) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsGradient_UT, InferShapeFakeQuantWithMinMaxVarsGradient_009) {
  ge::op::FakeQuantWithMinMaxVarsGradient op;
  op.UpdateInputDesc("gradients", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x", create_desc({4, 3, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}