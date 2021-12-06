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
#include "rnn.h"
#include "op_proto_test_util.h"

class BasicLSTMCellWeightGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BasicLSTMCellWeightGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BasicLSTMCellWeightGradTest TearDown" << std::endl;
  }
};

TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_000) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("h", create_desc({4, 3, 1}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_001) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_INT8));
  op.UpdateInputDesc("h", create_desc({4, 3}, ge::DT_INT8));
  op.UpdateOutputDesc("dw", create_desc_with_ori({}, ge::DT_INT8,ge::FORMAT_FRACTAL_Z,{},ge::FORMAT_FRACTAL_Z));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_002) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("h", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dgate", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateOutputDesc("dw", create_desc_with_ori({}, ge::DT_FLOAT16,ge::FORMAT_HWCN,{},ge::FORMAT_HWCN));


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dw_desc = op.GetOutputDesc("dw");
  std::vector<int64_t> expected_output_shape_dw = {6,12};
  EXPECT_EQ(output_dw_desc.GetShape().GetDims(), expected_output_shape_dw);
  EXPECT_EQ(output_dw_desc.GetDataType(), ge::DT_FLOAT16);
  
  auto output_db_desc = op.GetOutputDesc("db");
  std::vector<int64_t> expected_output_shape_db = {12};
  EXPECT_EQ(output_db_desc.GetShape().GetDims(), expected_output_shape_db);
  EXPECT_EQ(output_db_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_003) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("h", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dgate", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateOutputDesc("dw", create_desc_with_ori({}, ge::DT_FLOAT16,ge::FORMAT_ND,{},ge::FORMAT_ND));


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dw_desc = op.GetOutputDesc("dw");
  std::vector<int64_t> expected_output_shape_dw = {6,12};
  EXPECT_EQ(output_dw_desc.GetShape().GetDims(), expected_output_shape_dw);
  EXPECT_EQ(output_dw_desc.GetDataType(), ge::DT_FLOAT16);
  
  auto output_db_desc = op.GetOutputDesc("db");
  std::vector<int64_t> expected_output_shape_db = {12};
  EXPECT_EQ(output_db_desc.GetShape().GetDims(), expected_output_shape_db);
  EXPECT_EQ(output_db_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_004) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("h", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dgate", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateOutputDesc("dw", create_desc_with_ori({}, ge::DT_FLOAT16,ge::FORMAT_NCHW,{},ge::FORMAT_NCHW));


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dw_desc = op.GetOutputDesc("dw");
  std::vector<int64_t> expected_output_shape_dw = {12,6,1,1};
  EXPECT_EQ(output_dw_desc.GetShape().GetDims(), expected_output_shape_dw);
  EXPECT_EQ(output_dw_desc.GetDataType(), ge::DT_FLOAT16);
  
  auto output_db_desc = op.GetOutputDesc("db");
  std::vector<int64_t> expected_output_shape_db = {12,1,1,1};
  EXPECT_EQ(output_db_desc.GetShape().GetDims(), expected_output_shape_db);
  EXPECT_EQ(output_db_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(BasicLSTMCellWeightGradTest, InfershapeBasicLSTMCellWeightGrad_005) {
  ge::op::BasicLSTMCellWeightGrad op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("h", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dgate", create_desc({4, 3}, ge::DT_FLOAT16));
  op.UpdateOutputDesc("dw", create_desc_with_ori({}, ge::DT_FLOAT16,ge::FORMAT_NHWC,{},ge::FORMAT_NHWC));


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dw_desc = op.GetOutputDesc("dw");
  std::vector<int64_t> expected_output_shape_dw = {12,1,1,6};
  EXPECT_EQ(output_dw_desc.GetShape().GetDims(), expected_output_shape_dw);
  EXPECT_EQ(output_dw_desc.GetDataType(), ge::DT_FLOAT16);
  
  auto output_db_desc = op.GetOutputDesc("db");
  std::vector<int64_t> expected_output_shape_db = {12,1,1,1};
  EXPECT_EQ(output_db_desc.GetShape().GetDims(), expected_output_shape_db);
  EXPECT_EQ(output_db_desc.GetDataType(), ge::DT_FLOAT16);
}