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

class ProdVirialSeAProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdVirialSeA Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdVirialSeA Proto Test TearDown" << std::endl;
  }
};

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_1) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({1, 5087232}, ge::DT_FLOAT));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_2) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({2, 5087232}, ge::DT_FLOAT));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_3) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({1, 5087232}, ge::DT_FLOAT16));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({4}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAInferShapeTest_1) {
  int32_t nframes = 1;
  int32_t nloc = 12288;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;
  int32_t nall = 28328;
  int32_t natomsSize = 4;

  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc_shape_range({nframes, nloc * nnei * 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                          {nframes, nloc * nnei * 4}, ge::FORMAT_ND,
                                                          {{nframes, nframes}, {nloc * nnei * 4, nloc * nnei * 4}}));
  op.UpdateInputDesc("in_deriv",
                     create_desc_shape_range({nframes, nloc * nnei * 4 * 3}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {nframes, nloc * nnei * 4 * 3}, ge::FORMAT_ND,
                                             {{nframes, nframes}, {nloc * nnei * 4 * 3, nloc * nnei * 4 * 3}}));
  op.UpdateInputDesc("rij", create_desc_shape_range({nframes, nloc * nnei * 3}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                    {nframes, nloc * nnei * 3}, ge::FORMAT_ND,
                                                    {{nframes, nframes}, {nloc * nnei * 3, nloc * nnei * 3}}));
  op.UpdateInputDesc(
      "nlist", create_desc_shape_range({nframes, nloc * nnei}, ge::DT_INT32, ge::FORMAT_ND, {nframes, nloc * nnei},
                                       ge::FORMAT_ND, {{nframes, nframes}, {nloc * nnei, nloc * nnei}}));
  op.SetAttr("n_a_sel", n_a_sel);
  op.SetAttr("n_r_sel", n_r_sel);

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

  {
    auto output_desc = op.GetOutputDescByName("virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nframes, 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }

  {
    auto output_desc = op.GetOutputDescByName("atom_virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nframes, nall * 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAInferShapeTest_2) {
  int32_t nframes = 1;
  int32_t nloc = -1;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;
  int32_t nall = -1;
  int32_t natomsSize = 4;

  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc_shape_range({nframes, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nframes, -1},
                                                          ge::FORMAT_ND, {{nframes, nframes}, {1, -1}}));
  op.UpdateInputDesc("in_deriv", create_desc_shape_range({nframes, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nframes, -1},
                                                         ge::FORMAT_ND, {{nframes, nframes}, {1, -1}}));
  op.UpdateInputDesc("rij", create_desc_shape_range({nframes, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {nframes, -1},
                                                    ge::FORMAT_ND, {{nframes, nframes}, {1, -1}}));
  op.UpdateInputDesc("nlist", create_desc_shape_range({nframes, -1}, ge::DT_INT32, ge::FORMAT_ND, {nframes, -1},
                                                      ge::FORMAT_ND, {{nframes, nframes}, {1, -1}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({natomsSize}, ge::DT_INT32, ge::FORMAT_ND, {natomsSize},
                                                       ge::FORMAT_ND, {{natomsSize, natomsSize}}));
  op.SetAttr("n_a_sel", n_a_sel);
  op.SetAttr("n_r_sel", n_r_sel);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nframes, 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }

  {
    auto output_desc = op.GetOutputDescByName("atom_virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nframes, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{nframes, nframes}, {0, -1}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }
}
