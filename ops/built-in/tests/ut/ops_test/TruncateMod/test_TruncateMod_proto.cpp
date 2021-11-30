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

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class TruncateMod_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TruncateMod_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TruncateMod_UT TearDown" << std::endl;
  }
};

TEST_F(TruncateMod_UT, InferShapeTruncateMod_000) {
  ge::op::TruncateMod op;
  auto tensor_desc = create_desc({2, 100, 4}, ge::DT_FLOAT);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}