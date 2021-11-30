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
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class SPPTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SPPTest_UT test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SPPTest_UT test TearDown" << std::endl;
  }
};

TEST_F(SPPTest_UT, InferShapeSPP_000) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_001) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 2, 1}, ge::DT_INT8));
  op.SetAttr("pyramid_height", "zero");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_002) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 2, 1}, ge::DT_INT8));
  op.SetAttr("pyramid_height", 10);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_003) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 2, 1}, ge::DT_INT8));
  op.SetAttr("pyramid_height", 4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_004) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1024, 1024}, ge::DT_INT8));
  op.SetAttr("pyramid_height", 4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_005) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_006) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 2, 1}, ge::DT_INT8));
  op.SetAttr("pyramid_height", "zero");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SPPTest_UT, InferShapeSPP_007) {
  ge::op::SPP op;
  op.UpdateInputDesc("x", create_desc({4, 3, 256, 256}, ge::DT_FLOAT16));
  op.SetAttr("pyramid_height", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_box = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc_box.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_box = {4, 3, 1, 1};
  EXPECT_EQ(output_desc_box.GetShape().GetDims(), expected_output_shape_box);
}