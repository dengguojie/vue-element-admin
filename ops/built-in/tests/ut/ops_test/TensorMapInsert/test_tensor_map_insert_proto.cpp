/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include "map_ops.h"

class TensorMapInsert : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorMapInsert SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorMapInsert TearDown" << std::endl;
  }
};

TEST_F(TensorMapInsert, TensorMapInsert_infershape_test) {
  ge::op::TensorMapInsert op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("key", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4,2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto y_desc = op.GetOutputDescByName("output_handle");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_VARIANT);
}