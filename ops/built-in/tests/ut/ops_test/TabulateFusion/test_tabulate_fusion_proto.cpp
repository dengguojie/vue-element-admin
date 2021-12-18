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
#include "array_ops.h"
#include "deep_md.h"
#include "op_proto_test_util.h"

class TabulateFusionProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TabulateFusion Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TabulateFusion Proto Test TearDown" << std::endl;
  }
};

TEST_F(TabulateFusionProtoTest, TabulateFusionVerifyTest_1) {
  ge::op::TabulateFusion op;
  int64_t nloc = 4096;
  int64_t nnei = 46;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;
  op.UpdateInputDesc("table", create_desc({table_dim0, last_layer_size * 6}, ge::DT_FLOAT));
  op.UpdateInputDesc("table_info", create_desc({6}, ge::DT_FLOAT));
  op.UpdateInputDesc("em_x", create_desc({nloc * nnei, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("em", create_desc({nloc, nnei, 4}, ge::DT_FLOAT));
  op.SetAttr("last_layer_size", last_layer_size);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(TabulateFusionProtoTest, TabulateFusionVerifyTest_2) {
  ge::op::TabulateFusion op;
  int64_t nloc = 4096;
  int64_t nnei = 46;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;
  op.UpdateInputDesc("table", create_desc({table_dim0, last_layer_size * 6}, ge::DT_FLOAT));
  op.UpdateInputDesc("table_info", create_desc({6}, ge::DT_FLOAT));
  op.UpdateInputDesc("em_x", create_desc({nloc * nnei, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("em", create_desc({nloc, nnei, 4}, ge::DT_FLOAT));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(TabulateFusionProtoTest, TabulateFusionVerifyTest_3) {
  ge::op::TabulateFusion op;
  int64_t nloc = 4096;
  int64_t nnei = 46;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;
  op.UpdateInputDesc("table", create_desc({table_dim0}, ge::DT_FLOAT));
  op.UpdateInputDesc("table_info", create_desc({6}, ge::DT_FLOAT));
  op.UpdateInputDesc("em_x", create_desc({nloc * nnei, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("em", create_desc({nloc, nnei, 4}, ge::DT_FLOAT));
  op.SetAttr("last_layer_size", last_layer_size);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}


TEST_F(TabulateFusionProtoTest, TabulateFusionInferShapeTest_1) {
  ge::op::TabulateFusion op;
  int64_t nloc = -1;
  int64_t nnei = -1;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;

  op.UpdateInputDesc("table",
                     create_desc_shape_range({table_dim0, last_layer_size * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {table_dim0, last_layer_size * 6}, ge::FORMAT_ND,
                                             {{table_dim0, table_dim0}, {last_layer_size * 6, last_layer_size * 6}}));
  op.UpdateInputDesc("table_info",
                     create_desc_shape_range({6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {6}, ge::FORMAT_ND,
                                             {{6, 6}}));
  op.UpdateInputDesc("em_x",
                     create_desc_shape_range({nloc * nnei, 1}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {nloc * nnei, 1}, ge::FORMAT_ND,
                                             {{4096 * 46, 8192 * 92}, {1, 1}}));
  op.UpdateInputDesc("em",
                     create_desc_shape_range({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {nloc, nnei, 4}, ge::FORMAT_ND,
                                             {{4096, 8192}, {46, 92}, {4, 4}}));
  op.SetAttr("last_layer_size", last_layer_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("descriptor");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, 4, last_layer_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}

TEST_F(TabulateFusionProtoTest, TabulateFusionInferShapeTest_2) {
  ge::op::TabulateFusion op;
  int64_t nloc = 8192;
  int64_t nnei = 46;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;

  op.UpdateInputDesc("table",
                     create_desc_with_ori({table_dim0, last_layer_size * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {table_dim0, last_layer_size * 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("table_info",
                     create_desc_with_ori({6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::FORMAT_ND));
  op.UpdateInputDesc("em_x",
                     create_desc_with_ori({nloc * nnei, 1}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc * nnei, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("em",
                     create_desc_with_ori({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc, nnei, 4}, ge::FORMAT_ND));
  op.SetAttr("last_layer_size", last_layer_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("descriptor");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, 4, last_layer_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}

TEST_F(TabulateFusionProtoTest, TabulateFusionInferShapeTest_3) {
  ge::op::TabulateFusion op;
  int64_t nloc = 8192;
  int64_t nnei = 92;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;
  int32_t splitCount = 2;
  int32_t splitIndex = 0;

  op.UpdateInputDesc("table",
                     create_desc_with_ori({table_dim0, last_layer_size * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {table_dim0, last_layer_size * 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("table_info",
                     create_desc_with_ori({6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::FORMAT_ND));
  op.UpdateInputDesc("em_x",
                     create_desc_with_ori({nloc * nnei, 1}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc * nnei, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("em",
                     create_desc_with_ori({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc, nnei, 4}, ge::FORMAT_ND));
  op.SetAttr("last_layer_size", last_layer_size);
  op.SetAttr("split_count", splitCount);
  op.SetAttr("split_index", splitIndex);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("descriptor");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {4096, 4, last_layer_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}

TEST_F(TabulateFusionProtoTest, TabulateFusionInferShapeTest_4) {
  ge::op::TabulateFusion op;
  int64_t nloc = 8192;
  int64_t nnei = 92;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 1360;
  int32_t splitCount = 2;
  int32_t splitIndex = 1;

  op.UpdateInputDesc("table",
                     create_desc_with_ori({table_dim0, last_layer_size * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {table_dim0, last_layer_size * 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("table_info",
                     create_desc_with_ori({6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::FORMAT_ND));
  op.UpdateInputDesc("em_x",
                     create_desc_with_ori({nloc * nnei, 1}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc * nnei, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("em",
                     create_desc_with_ori({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                          {nloc, nnei, 4}, ge::FORMAT_ND));
  op.SetAttr("last_layer_size", last_layer_size);
  op.SetAttr("split_count", splitCount);
  op.SetAttr("split_index", splitIndex);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("descriptor");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {4096, 4, last_layer_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}
