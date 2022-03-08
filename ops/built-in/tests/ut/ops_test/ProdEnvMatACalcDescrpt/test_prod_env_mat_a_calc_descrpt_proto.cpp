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

class ProdEnvMatACalcDescrptProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdEnvMatACalcDescrpt Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdEnvMatACalcDescrpt Proto Test TearDown" << std::endl;
  }
};

TEST_F(ProdEnvMatACalcDescrptProtoTest, ProdEnvMatACalcDescrptVerifyTest_0) {
  int32_t nsample = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;

  ge::op::ProdEnvMatACalcDescrpt op;
  op.UpdateInputDesc("distance", create_desc({nsample, nloc * nnei}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij_x", create_desc({nsample, nloc * nnei}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij_y", create_desc({nsample, nloc * nnei}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij_z", create_desc({nsample, nloc * nnei}, ge::DT_FLOAT));
  op.UpdateInputDesc("type", create_desc({nsample, nloc}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4, }, ge::DT_INT32));
  op.UpdateInputDesc("mesh", create_desc({1 + 1026 * nloc}, ge::DT_INT32));
  op.UpdateInputDesc("davg", create_desc({2, nnei * 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("dstd", create_desc({2, nnei * 4}, ge::DT_FLOAT));
  op.SetAttr("rcut_a", float(0.0));
  op.SetAttr("rcut_r", float(8.0));
  op.SetAttr("rcut_r_smth", float(2.0));
  op.SetAttr("sel_a", std::vector<int32_t>{46, 92});
  op.SetAttr("sel_r", std::vector<int32_t>{});
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ProdEnvMatACalcDescrptProtoTest, ProdEnvMatACalcDescrptInferShapeTest_1) {
  int32_t nsample = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;

  ge::op::ProdEnvMatACalcDescrpt op;
  op.UpdateInputDesc(
      "distance", create_desc_shape_range({nsample, nloc * nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, nloc * nnei},
                                          ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc(
      "rij_x", create_desc_shape_range({nsample, nloc * nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, nloc * nnei},
                                       ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc(
      "rij_y", create_desc_shape_range({nsample, nloc * nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, nloc * nnei},
                                       ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc(
      "rij_z", create_desc_shape_range({nsample, nloc * nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, nloc * nnei},
                                       ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc("type", create_desc_shape_range({nsample, nloc}, ge::DT_INT32, ge::FORMAT_ND, {nsample, nloc},
                                                     ge::FORMAT_ND, {{nsample, nloc}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND, {{4, 4}}));
  op.UpdateInputDesc("mesh", create_desc_shape_range({1 + nloc * 1026}, ge::DT_INT32, ge::FORMAT_ND, {1 + nloc * 1026},
                                                     ge::FORMAT_ND, {{1 + nloc * 1026, 1 + nloc * 1026}}));
  op.UpdateInputDesc("davg", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, nnei * 4},
                                                     ge::FORMAT_ND, {{2, 2}, {nnei * 4, nnei * 4}}));
  op.UpdateInputDesc("dstd", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, nnei * 4},
                                                     ge::FORMAT_ND, {{2, 2}, {nnei * 4, nnei * 4}}));
  op.SetAttr("rcut_a", float(0.0));
  op.SetAttr("rcut_r", float(8.0));
  op.SetAttr("rcut_r_smth", float(2.0));
  op.SetAttr("sel_a", std::vector<int32_t>{46, 92});
  op.SetAttr("sel_r", std::vector<int32_t>{});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ProdEnvMatACalcDescrptProtoTest, ProdEnvMatACalcDescrptInferShapeTest_2) {
  int32_t nsample = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;

  ge::op::ProdEnvMatACalcDescrpt op;
  op.UpdateInputDesc("distance",
                     create_desc_shape_range({nsample, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, -1}, ge::FORMAT_ND,
                                             {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc("rij_x", create_desc_shape_range({nsample, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, -1},
                                                      ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc("rij_y", create_desc_shape_range({nsample, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, -1},
                                                      ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc("rij_z", create_desc_shape_range({nsample, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nsample, -1},
                                                      ge::FORMAT_ND, {{nsample, nsample}, {nloc * nnei, nloc * nnei}}));
  op.UpdateInputDesc("type", create_desc_shape_range({nsample, -1}, ge::DT_INT32, ge::FORMAT_ND, {nsample, -1},
                                                      ge::FORMAT_ND, {{nsample, -1}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND, {{4, 4}}));
  op.UpdateInputDesc("mesh", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND,
                                                     {{1 + nloc * 1026, 1 + nloc * 1026}}));
  op.UpdateInputDesc("davg", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, nnei * 4},
                                                     ge::FORMAT_ND, {{2, 2}, {nnei * 4, nnei * 4}}));
  op.UpdateInputDesc("dstd", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, nnei * 4},
                                                     ge::FORMAT_ND, {{2, 2}, {nnei * 4, nnei * 4}}));
  op.SetAttr("rcut_a", float(0.0));
  op.SetAttr("rcut_r", float(8.0));
  op.SetAttr("rcut_r_smth", float(2.0));
  op.SetAttr("sel_a", std::vector<int32_t>{46, 92});
  op.SetAttr("sel_r", std::vector<int32_t>{});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
