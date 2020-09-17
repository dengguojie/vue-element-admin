/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*
 * \file test_max_pool_v3_proto.cpp
 * \brief ut of max pool v3
 */
#include <gtest/gtest.h>
#include <vector>
#include "max_pool_v3.h"

class MaxPoolV3Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "max_pool_v3 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "max_pool_v3 test TearDown" << std::endl;
  }
};

TEST_F(MaxPoolV3Test, max_pool_v3_test_case_1) {
  //  define your op here
  ge::op::MaxPoolV3 max_pool_v3_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({1, 64, 56, 56});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);

  //  update op input here
  max_pool_v3_op.UpdateInputDesc("x", tensorDesc);

  //  call InferShapeAndType function here
  auto ret = max_pool_v3_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = max_pool_v3_op.GetOutputDesc("output_data");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 64, 28, 28};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
