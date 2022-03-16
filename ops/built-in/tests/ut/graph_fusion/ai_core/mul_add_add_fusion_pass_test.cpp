/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "framework/common/types.h"
#include "split_combination_ops.h"
#define private public
#define protected public
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;
using namespace std;

class mul_add_add_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mul_add_add_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mul_add_add_fusion_test TearDown" << std::endl;
  }
};

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
  ge::GeTensorDesc desc_##name(shape_out, format_out, dtype);              \
  desc_##name.SetOriginFormat(format_in);                                  \
  desc_##name.SetOriginShape(shape_in)

TEST_F(mul_add_add_fusion_test, mul_add_add_fusion_test_1) {
  ge::Graph graph("mul_add_add_fusion_test_1");
  DESC_DATA(data_a, ge::GeShape({640, 1}), FORMAT_ND, ge::GeShape({640, 1}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_concat, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_const, ge::GeShape({4620}), FORMAT_ND, ge::GeShape({4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(trans_b, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(mul, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_1, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_2, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(trans_a, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);

  ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
  ge::OpDescPtr data_concat = std::make_shared<ge::OpDesc>("data_b", "ConcatV2D");
  ge::OpDescPtr data_const = std::make_shared<ge::OpDesc>("data_const", "Constant");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  ge::OpDescPtr add_1 = std::make_shared<ge::OpDesc>("add_1", "Add");
  ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
  ge::OpDescPtr add_2 = std::make_shared<ge::OpDesc>("add_2", "Add");
  ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  data_a->AddOutputDesc(desc_data_a);
  data_concat->AddOutputDesc(desc_data_concat);

  ge::GeTensorPtr assitPtr = nullptr;
  int32_t num = 4620;
  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(FORMAT_ND);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  assitPtr =
      std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float));
  ge::AttrUtils::SetTensor(data_const, "value", assitPtr);
  data_const->AddOutputDesc(desc_data_const);

  trans_b->AddInputDesc(desc_data_concat);
  trans_b->AddOutputDesc(desc_trans_b);
  ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_b, "dst_format", "FORMAT_FRACTAL_NZ");

  mul->AddInputDesc("x1", desc_data_a);
  mul->AddInputDesc("x2", desc_data_concat);
  mul->AddOutputDesc(desc_mul);
  add_1->AddInputDesc("x1", desc_mul);
  add_1->AddInputDesc("x2", desc_data_const);
  add_1->AddOutputDesc(desc_add_1);
  trans_a->AddInputDesc(desc_add_1);
  trans_a->AddOutputDesc(desc_trans_a);
  ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_a, "dst_format", "FORMAT_FRACTAL_NZ");
  add_2->AddInputDesc("x1", desc_trans_b);
  add_2->AddInputDesc("x2", desc_trans_a);
  add_2->AddOutputDesc(desc_add_2);
  netoutput->AddInputDesc(desc_add_2);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("mul_add_add_graph");
  ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
  ge::NodePtr data_concat_node = compute_graph_ptr->AddNode(data_concat);
  ge::NodePtr data_const_node = compute_graph_ptr->AddNode(data_const);
  ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
  ge::NodePtr mul_node = compute_graph_ptr->AddNode(mul);
  ge::NodePtr add_1_node = compute_graph_ptr->AddNode(add_1);
  ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
  ge::NodePtr add_2_node = compute_graph_ptr->AddNode(add_2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_const_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_1_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(add_2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status status = 
  fe::FusionPassTestUtils::RunGraphFusionPass("ZMulAddAddFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                                           *compute_graph_ptr);
  EXPECT_EQ(status, fe::SUCCESS);
}


TEST_F(mul_add_add_fusion_test, mul_add_add_fusion_test_2) {
  ge::Graph graph("mul_add_add_fusion_test_2");
  DESC_DATA(data_a, ge::GeShape({630, 1}), FORMAT_ND, ge::GeShape({630, 1}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_concat, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_const, ge::GeShape({4620}), FORMAT_ND, ge::GeShape({4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(trans_b, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(mul, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_1, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_2, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(trans_a, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);

  ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
  ge::OpDescPtr data_concat = std::make_shared<ge::OpDesc>("data_b", "ConcatV2D");
  ge::OpDescPtr data_const = std::make_shared<ge::OpDesc>("data_const", "Constant");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  ge::OpDescPtr add_1 = std::make_shared<ge::OpDesc>("add_1", "Add");
  ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
  ge::OpDescPtr add_2 = std::make_shared<ge::OpDesc>("add_2", "Add");
  ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  data_a->AddOutputDesc(desc_data_a);
  data_concat->AddOutputDesc(desc_data_concat);

  ge::GeTensorPtr assitPtr = nullptr;
  int32_t num = 4620;
  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(FORMAT_ND);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  assitPtr =
      std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float));
  ge::AttrUtils::SetTensor(data_const, "value", assitPtr);
  data_const->AddOutputDesc(desc_data_const);

  trans_b->AddInputDesc(desc_data_concat);
  trans_b->AddOutputDesc(desc_trans_b);
  ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_b, "dst_format", "FORMAT_FRACTAL_NZ");

  mul->AddInputDesc("x1", desc_data_a);
  mul->AddInputDesc("x2", desc_data_concat);
  mul->AddOutputDesc(desc_mul);
  add_1->AddInputDesc("x1", desc_mul);
  add_1->AddInputDesc("x2", desc_data_const);
  add_1->AddOutputDesc(desc_add_1);
  trans_a->AddInputDesc(desc_add_1);
  trans_a->AddOutputDesc(desc_trans_a);
  ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_a, "dst_format", "FORMAT_FRACTAL_NZ");
  add_2->AddInputDesc("x1", desc_trans_b);
  add_2->AddInputDesc("x2", desc_trans_a);
  add_2->AddOutputDesc(desc_add_2);
  netoutput->AddInputDesc(desc_add_2);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("mul_add_add_graph");
  ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
  ge::NodePtr data_concat_node = compute_graph_ptr->AddNode(data_concat);
  ge::NodePtr data_const_node = compute_graph_ptr->AddNode(data_const);
  ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
  ge::NodePtr mul_node = compute_graph_ptr->AddNode(mul);
  ge::NodePtr add_1_node = compute_graph_ptr->AddNode(add_1);
  ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
  ge::NodePtr add_2_node = compute_graph_ptr->AddNode(add_2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_const_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_1_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(add_2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status status =
  fe::FusionPassTestUtils::RunGraphFusionPass("ZMulAddAddFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                                           *compute_graph_ptr);
  EXPECT_EQ(status, fe::SUCCESS);
}


TEST_F(mul_add_add_fusion_test, mul_add_add_fusion_test_3) {
  ge::Graph graph("mul_add_add_fusion_test_3");
  DESC_DATA(data_a, ge::GeShape({630, 1}), FORMAT_FRACTAL_NZ, ge::GeShape({630, 1}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(data_concat, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_const, ge::GeShape({4620}), FORMAT_ND, ge::GeShape({4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(trans_b, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(mul, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_1, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_2, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(trans_a, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);

  ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
  ge::OpDescPtr data_concat = std::make_shared<ge::OpDesc>("data_b", "ConcatV2D");
  ge::OpDescPtr data_const = std::make_shared<ge::OpDesc>("data_const", "Constant");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  ge::OpDescPtr add_1 = std::make_shared<ge::OpDesc>("add_1", "Add");
  ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
  ge::OpDescPtr add_2 = std::make_shared<ge::OpDesc>("add_2", "Add");
  ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  data_a->AddOutputDesc(desc_data_a);
  data_concat->AddOutputDesc(desc_data_concat);

  ge::GeTensorPtr assitPtr = nullptr;
  int32_t num = 4620;
  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(FORMAT_ND);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  assitPtr =
      std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float));
  ge::AttrUtils::SetTensor(data_const, "value", assitPtr);
  data_const->AddOutputDesc(desc_data_const);

  trans_b->AddInputDesc(desc_data_concat);
  trans_b->AddOutputDesc(desc_trans_b);
  ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_b, "dst_format", "FORMAT_FRACTAL_NZ");

  mul->AddInputDesc("x1", desc_data_a);
  mul->AddInputDesc("x2", desc_data_concat);
  mul->AddOutputDesc(desc_mul);
  add_1->AddInputDesc("x1", desc_mul);
  add_1->AddInputDesc("x2", desc_data_const);
  add_1->AddOutputDesc(desc_add_1);
  trans_a->AddInputDesc(desc_add_1);
  trans_a->AddOutputDesc(desc_trans_a);
  ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_a, "dst_format", "FORMAT_FRACTAL_NZ");
  add_2->AddInputDesc("x1", desc_trans_b);
  add_2->AddInputDesc("x2", desc_trans_a);
  add_2->AddOutputDesc(desc_add_2);
  netoutput->AddInputDesc(desc_add_2);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("mul_add_add_graph");
  ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
  ge::NodePtr data_concat_node = compute_graph_ptr->AddNode(data_concat);
  ge::NodePtr data_const_node = compute_graph_ptr->AddNode(data_const);
  ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
  ge::NodePtr mul_node = compute_graph_ptr->AddNode(mul);
  ge::NodePtr add_1_node = compute_graph_ptr->AddNode(add_1);
  ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
  ge::NodePtr add_2_node = compute_graph_ptr->AddNode(add_2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_const_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_1_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(add_2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status status =
  fe::FusionPassTestUtils::RunGraphFusionPass("ZMulAddAddFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                                           *compute_graph_ptr);
  EXPECT_EQ(status, fe::NOT_CHANGED);
}

TEST_F(mul_add_add_fusion_test, mul_add_add_fusion_test_4) {
  ge::Graph graph("mul_add_add_fusion_test_4");
  DESC_DATA(data_a, ge::GeShape({630, 1}), FORMAT_ND, ge::GeShape({630, 1}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_concat, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_const, ge::GeShape({16}), FORMAT_ND, ge::GeShape({16}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(trans_b, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(mul, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_1, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_2, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(trans_a, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);

  ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
  ge::OpDescPtr data_concat = std::make_shared<ge::OpDesc>("data_b", "ConcatV2D");
  ge::OpDescPtr data_const = std::make_shared<ge::OpDesc>("data_const", "Constant");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  ge::OpDescPtr add_1 = std::make_shared<ge::OpDesc>("add_1", "Add");
  ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
  ge::OpDescPtr add_2 = std::make_shared<ge::OpDesc>("add_2", "Add");
  ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  data_a->AddOutputDesc(desc_data_a);
  data_concat->AddOutputDesc(desc_data_concat);

  ge::GeTensorPtr assitPtr = nullptr;
  int32_t num = 16;
  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(FORMAT_ND);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  assitPtr =
      std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float));
  ge::AttrUtils::SetTensor(data_const, "value", assitPtr);
  data_const->AddOutputDesc(desc_data_const);

  trans_b->AddInputDesc(desc_data_concat);
  trans_b->AddOutputDesc(desc_trans_b);
  ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_b, "dst_format", "FORMAT_FRACTAL_NZ");

  mul->AddInputDesc("x1", desc_data_a);
  mul->AddInputDesc("x2", desc_data_concat);
  mul->AddOutputDesc(desc_mul);
  add_1->AddInputDesc("x1", desc_mul);
  add_1->AddInputDesc("x2", desc_data_const);
  add_1->AddOutputDesc(desc_add_1);
  trans_a->AddInputDesc(desc_add_1);
  trans_a->AddOutputDesc(desc_trans_a);
  ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_a, "dst_format", "FORMAT_FRACTAL_NZ");
  add_2->AddInputDesc("x1", desc_trans_b);
  add_2->AddInputDesc("x2", desc_trans_a);
  add_2->AddOutputDesc(desc_add_2);
  netoutput->AddInputDesc(desc_add_2);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("mul_add_add_graph");
  ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
  ge::NodePtr data_concat_node = compute_graph_ptr->AddNode(data_concat);
  ge::NodePtr data_const_node = compute_graph_ptr->AddNode(data_const);
  ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
  ge::NodePtr mul_node = compute_graph_ptr->AddNode(mul);
  ge::NodePtr add_1_node = compute_graph_ptr->AddNode(add_1);
  ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
  ge::NodePtr add_2_node = compute_graph_ptr->AddNode(add_2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_const_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_1_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(add_2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status status =
  fe::FusionPassTestUtils::RunGraphFusionPass("ZMulAddAddFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                                           *compute_graph_ptr);
  EXPECT_EQ(status, fe::NOT_CHANGED);
}

TEST_F(mul_add_add_fusion_test, mul_add_add_fusion_test_5) {
  ge::Graph graph("mul_add_add_fusion_test_5");
  DESC_DATA(data_a, ge::GeShape({630, 2}), FORMAT_ND, ge::GeShape({630, 2}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_concat, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(data_const, ge::GeShape({4620}), FORMAT_ND, ge::GeShape({4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(trans_b, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(mul, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_1, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({640, 4620}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(add_2, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, ge::GeShape({640, 4620}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  DESC_DATA(trans_a, ge::GeShape({640, 4620}), FORMAT_ND, ge::GeShape({289, 40, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);

  ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
  ge::OpDescPtr data_concat = std::make_shared<ge::OpDesc>("data_b", "ConcatV2D");
  ge::OpDescPtr data_const = std::make_shared<ge::OpDesc>("data_const", "Constant");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  ge::OpDescPtr add_1 = std::make_shared<ge::OpDesc>("add_1", "Add");
  ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
  ge::OpDescPtr add_2 = std::make_shared<ge::OpDesc>("add_2", "Add");
  ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  data_a->AddOutputDesc(desc_data_a);
  data_concat->AddOutputDesc(desc_data_concat);

  ge::GeTensorPtr assitPtr = nullptr;
  int32_t num = 4620;
  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(FORMAT_ND);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  assitPtr =
      std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float));
  ge::AttrUtils::SetTensor(data_const, "value", assitPtr);
  data_const->AddOutputDesc(desc_data_const);

  trans_b->AddInputDesc(desc_data_concat);
  trans_b->AddOutputDesc(desc_trans_b);
  ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_b, "dst_format", "FORMAT_FRACTAL_NZ");

  mul->AddInputDesc("x1", desc_data_a);
  mul->AddInputDesc("x2", desc_data_concat);
  mul->AddOutputDesc(desc_mul);
  add_1->AddInputDesc("x1", desc_mul);
  add_1->AddInputDesc("x2", desc_data_const);
  add_1->AddOutputDesc(desc_add_1);
  trans_a->AddInputDesc(desc_add_1);
  trans_a->AddOutputDesc(desc_trans_a);
  ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
  ge::AttrUtils::SetStr(trans_a, "dst_format", "FORMAT_FRACTAL_NZ");
  add_2->AddInputDesc("x1", desc_trans_b);
  add_2->AddInputDesc("x2", desc_trans_a);
  add_2->AddOutputDesc(desc_add_2);
  netoutput->AddInputDesc(desc_add_2);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("mul_add_add_graph");
  ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
  ge::NodePtr data_concat_node = compute_graph_ptr->AddNode(data_concat);
  ge::NodePtr data_const_node = compute_graph_ptr->AddNode(data_const);
  ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
  ge::NodePtr mul_node = compute_graph_ptr->AddNode(mul);
  ge::NodePtr add_1_node = compute_graph_ptr->AddNode(add_1);
  ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
  ge::NodePtr add_2_node = compute_graph_ptr->AddNode(add_2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_const_node->GetOutDataAnchor(0), add_1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_1_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_concat_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), add_2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(add_2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status status =
  fe::FusionPassTestUtils::RunGraphFusionPass("ZMulAddAddFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                                           *compute_graph_ptr);
  EXPECT_EQ(status, fe::NOT_CHANGED);
}