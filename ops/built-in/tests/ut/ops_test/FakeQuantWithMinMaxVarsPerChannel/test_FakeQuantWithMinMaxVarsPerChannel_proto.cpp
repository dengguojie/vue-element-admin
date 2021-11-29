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

class FakeQuantWithMinMaxVarsPerChannel_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FakeQuantWithMinMaxVarsPerChannel_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FakeQuantWithMinMaxVarsPerChannel_UT TearDown" << std::endl;
  }
};

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_001) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
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

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_002) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  std::string num_bits = "20";
  op.SetAttr("num_bits", num_bits);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_003) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
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

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_004) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 20;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_005) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_006) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_007) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannel_UT, InferShapeFakeQuantWithMinMaxVarsPerChannel_008) {
  ge::op::FakeQuantWithMinMaxVarsPerChannel op;
  op.UpdateInputDesc("x", create_desc({4, 3, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT16));
  int8_t num_bits = 8;
  op.SetAttr("num_bits", num_bits);
  bool narrow_range = false;
  op.SetAttr("narrow_range", narrow_range);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}