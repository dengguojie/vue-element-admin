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

class BiasAdd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasAdd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasAdd TearDown" << std::endl;
  }
};

TEST_F(BiasAdd, VerifyBiasAdd_001) {
  ge::op::BiasAdd op;
  op.SetAttr("data_format", false);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BiasAdd, VerifyBiasAdd_002) {
  ge::op::BiasAdd op;
  op.SetAttr("data_format", "ND");
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}