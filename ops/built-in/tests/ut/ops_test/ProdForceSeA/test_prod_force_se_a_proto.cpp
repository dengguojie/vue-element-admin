/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_prod_force_se_a_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "deep_md.h"

class ProdForceSeAProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdForceSeA Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdForceSeA Proto Test TearDown" << std::endl;
  }
};

TEST_F(ProdForceSeAProtoTest, ProdForceSeAVerifyTest_1) {
  ge::op::ProdForceSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ProdForceSeAProtoTest, ProdForceSeAInferShapeTest_1) {
  ge::op::ProdForceSeA op;
  op.UpdateInputDesc("net_deriv", create_desc_shape_range({1, 6782976}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 6782976},
                                                          ge::FORMAT_ND, {{1, 1}, {6782976, 6782976}}));
  op.UpdateInputDesc("in_deriv", create_desc_shape_range({1, 20348928}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 20348928},
                                                         ge::FORMAT_ND, {{1, 1}, {20348928, 20348928}}));
  op.UpdateInputDesc("nlist", create_desc_shape_range({1, 1695744}, ge::DT_INT32, ge::FORMAT_ND, {1, 1695744},
                                                      ge::FORMAT_ND, {{1, 1}, {1695744, 1695744}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({3}, ge::DT_INT32, ge::FORMAT_ND, {3}, ge::FORMAT_ND,
                                                       {{3, 3}}));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("atom_force");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {1, -1, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 1}, {0, -1}, {3, 3}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }
}

TEST_F(ProdForceSeAProtoTest, ProdForceSeAInferShapeTest_2) {
  ge::op::ProdForceSeA op;
  op.UpdateInputDesc("net_deriv", create_desc_shape_range({1, 6782976}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 6782976},
                                                          ge::FORMAT_ND, {{1, 1}, {6782976, 6782976}}));
  op.UpdateInputDesc("in_deriv", create_desc_shape_range({1, 20348928}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 20348928},
                                                         ge::FORMAT_ND, {{1, 1}, {20348928, 20348928}}));
  op.UpdateInputDesc("nlist", create_desc_shape_range({1, 1695744}, ge::DT_INT32, ge::FORMAT_ND, {1, 1695744},
                                                      ge::FORMAT_ND, {{1, 1}, {1695744, 1695744}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({3}, ge::DT_INT32, ge::FORMAT_ND, {3}, ge::FORMAT_ND,
                                                       {{3, 3}}));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  op.SetAttr("second_infer", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("atom_force");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {1, -1, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 1}, {0, -1}, {3, 3}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }
}
