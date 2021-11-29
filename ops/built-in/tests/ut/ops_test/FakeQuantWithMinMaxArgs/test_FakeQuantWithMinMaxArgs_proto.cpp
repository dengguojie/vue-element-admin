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

class FakeQuantWithMinMaxArgs_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FakeQuantWithMinMaxArgs_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FakeQuantWithMinMaxArgs_UT TearDown" << std::endl;
  }
};

TEST_F(FakeQuantWithMinMaxArgs_UT, InferShapeFakeQuantWithMinMaxArgs_001) {
  ge::op::FakeQuantWithMinMaxArgs op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  float min = -6.0;
  float max = 6.0;
  int8_t num_bits = 8;
  bool narrow_range = false;
  op.SetAttr("min", min);
  op.SetAttr("max", max);
  op.SetAttr("num_bits", num_bits);
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(FakeQuantWithMinMaxArgs_UT, InferShapeFakeQuantWithMinMaxArgs_002) {
  ge::op::FakeQuantWithMinMaxArgs op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  std::string min = "-6.0";
  op.SetAttr("min", min);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgs_UT, InferShapeFakeQuantWithMinMaxArgs_003) {
  ge::op::FakeQuantWithMinMaxArgs op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  std::string max = "-6.0";
  op.SetAttr("max", max);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgs_UT, InferShapeFakeQuantWithMinMaxArgs_004) {
  ge::op::FakeQuantWithMinMaxArgs op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  float min = 6.0;
  op.SetAttr("min", min);
  float max = -6.0;
  op.SetAttr("max", max);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
