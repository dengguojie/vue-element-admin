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

class ProdEnvMatAProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdEnvMatA Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdEnvMatA Proto Test TearDown" << std::endl;
  }
};

TEST_F(ProdEnvMatAProtoTest, ProdEnvMatAVerifyTest_0) {
  ge::op::ProdEnvMatA op;
  op.UpdateInputDesc("coord", create_desc({1, 84984}, ge::DT_FLOAT));
  op.UpdateInputDesc("type", create_desc({1, 28328}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("box", create_desc({1, 9}, ge::DT_FLOAT));
  op.UpdateInputDesc("mesh", create_desc({1 + 1026 * 12288}, ge::DT_INT32));
  op.UpdateInputDesc("davg", create_desc({2, 138 * 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("dstd", create_desc({2, 138 * 4}, ge::DT_FLOAT));
  op.SetAttr("rcut_a", float(0.0));
  op.SetAttr("rcut_r", float(8.0));
  op.SetAttr("rcut_r_smth", float(2.0));
  op.SetAttr("sel_a", std::vector<int32_t>{46, 92});
  op.SetAttr("sel_r", std::vector<int32_t>{});
  op.SetAttr("split_count", (int)2);
  op.SetAttr("split_index", (int)0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}


TEST_F(ProdEnvMatAProtoTest, ProdEnvMatAInferShapeTest_1) {
  int32_t nsample = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;
  int32_t nall = 28328;
  int32_t natomsSize = 4;

  ge::op::ProdEnvMatA op;
  op.UpdateInputDesc("coord", create_desc_shape_range({nsample, nall * 3}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {nsample, nall * 3}, ge::FORMAT_ND,
                                                          {{nsample, nsample}, {nall * 3, nall * 3}}));
  op.UpdateInputDesc("type",
                     create_desc_shape_range({nsample, nall}, ge::DT_INT32, ge::FORMAT_ND,
                                             {nsample, nall}, ge::FORMAT_ND,
                                             {{nsample, nsample}, {nall, nall}}));

  op.UpdateInputDesc("box",
                     create_desc_shape_range({nsample, 9}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {nsample, 9}, ge::FORMAT_ND,
                                             {{nsample, nsample}, {9, 9}}));
  op.UpdateInputDesc("mesh",
                     create_desc_shape_range({1 + nloc * 1026}, ge::DT_INT32, ge::FORMAT_ND,
                                             {1 + nloc * 1026}, ge::FORMAT_ND,
                                             {{1 + nloc * 1026, 1 + nloc * 1026}}));
  op.UpdateInputDesc("davg", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {2, nnei * 4}, ge::FORMAT_ND,
                                                          {{2, 2}, {nnei * 4, nnei * 4}}));
  op.UpdateInputDesc("dstd", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {2, nnei * 4}, ge::FORMAT_ND,
                                                          {{2, 2}, {nnei * 4, nnei * 4}}));
  op.SetAttr("rcut_a", float(0.0));
  op.SetAttr("rcut_r", float(8.0));
  op.SetAttr("rcut_r_smth", float(2.0));
  op.SetAttr("sel_a", std::vector<int32_t>{46, 92});
  op.SetAttr("sel_r", std::vector<int32_t>{});
  op.SetAttr("split_count", (int)2);
  op.SetAttr("split_index", (int)1);
  {
    ge::TensorDesc tensorDesc(ge::Shape({natomsSize}), ge::FORMAT_ND, ge::DT_INT32);
    int32_t tensorValue[natomsSize] = {nloc, nall, 0, 1};
    ge::Tensor natomsTensor = ge::Tensor(tensorDesc, (uint8_t*)tensorValue, natomsSize * sizeof(int32_t));

    auto natomsOp = ge::op::Const("natoms").set_attr_value(natomsTensor);
    natomsOp.UpdateOutputDesc("y", tensorDesc);

    op.UpdateInputDesc("natoms", tensorDesc);
    op.set_input_natoms(natomsOp);
  }

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ProdEnvMatAProtoTest, ProdEnvMatAInferShapeTest_2) {
  int32_t nsample = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;
  int32_t nall = 28328;
  int32_t natomsSize = 4;

  ge::op::ProdEnvMatA op;
  op.UpdateInputDesc("coord", create_desc_shape_range({nsample, -1}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {nsample, -1}, ge::FORMAT_ND,
                                                          {{nsample, nsample}, {nall * 3, nall * 3}}));
  op.UpdateInputDesc("type",
                     create_desc_shape_range({nsample, -1}, ge::DT_INT32, ge::FORMAT_ND,
                                             {nsample, -1}, ge::FORMAT_ND,
                                             {{nsample, nsample}, {nall, nall}}));
  op.UpdateInputDesc("natoms",
                     create_desc_shape_range({4}, ge::DT_INT32, ge::FORMAT_ND,
                                             {4}, ge::FORMAT_ND,
                                             {{4, 4}}));
  op.UpdateInputDesc("box",
                     create_desc_shape_range({nsample, 9}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {nsample, 9}, ge::FORMAT_ND,
                                             {{nsample, nsample}, {9, 9}}));
  op.UpdateInputDesc("mesh",
                     create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND,
                                             {-1}, ge::FORMAT_ND,
                                             {{1 + nloc * 1026, 1 + nloc * 1026}}));
  op.UpdateInputDesc("davg", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {2, nnei * 4}, ge::FORMAT_ND,
                                                          {{2, 2}, {nnei * 4, nnei * 4}}));
  op.UpdateInputDesc("dstd", create_desc_shape_range({2, nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {2, nnei * 4}, ge::FORMAT_ND,
                                                          {{2, 2}, {nnei * 4, nnei * 4}}));
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
