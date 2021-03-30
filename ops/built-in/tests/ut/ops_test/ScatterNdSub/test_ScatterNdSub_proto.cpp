/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "matrix_calculation_ops.h"

class scatter_nd_sub : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scatter_nd_sub SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scatter_nd_sub TearDown" << std::endl;
  }
};

TEST_F(scatter_nd_sub, scatter_nd_sub_infershape_test) {
  ge::op::ScatterNdSub op;
  op.UpdateInputDesc("var", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices",
                     create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("updates",
                     create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {33, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(scatter_nd_sub, scatter_nd_sub_verify_invalid_test) {
  ge::op::ScatterNdSub op;
  op.UpdateInputDesc("var", create_desc_with_ori({33, 5}, ge::DT_INT32, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices",
                     create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("updates",
                     create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
