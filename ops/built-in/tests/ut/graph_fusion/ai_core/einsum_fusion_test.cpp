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

#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"

#define private public
#define protected public

#include "graph_fusion/ai_core/einsum_fusion_pass.h"

using namespace ge;
using namespace op;

struct EinsumTestParam {
  size_t input_num;
  std::string case_name;
  std::string equation;
  std::string last_node_name;
  vector<std::pair<int64_t, int64_t>> input_x1_range;
  vector<std::pair<int64_t, int64_t>> input_x2_range;
  vector<std::pair<int64_t, int64_t>> output_range;
  std::map<std::string, uint32_t> expected_nodes;
};

class einsum_fusion_test : public testing::TestWithParam<EinsumTestParam> {
 private:
  ge::NodePtr GetNode(const std::string &node_name, const ge::ComputeGraphPtr &graph) {
    for (auto node : graph->GetAllNodes()) {
      if (node->GetName() == node_name) {
        return node;
      }
    }

    return nullptr;
  }

  bool GetShapeFromRange(const vector<std::pair<int64_t, int64_t>> &shape_range, std::vector<int64_t> &shape) {
    bool dynamic = false;
    shape.reserve(shape_range.size());
    for (auto &range: shape_range) {
      if (range.first == range.second) {
        shape.push_back(range.first);
      } else {
        dynamic = true;
        shape.push_back(-1);
      }
    }
    return dynamic;
  }
};

static EinsumTestParam einsum_general_cases_params[] = {
    {2,
     "static_shape_scene_1",
     "abc,cde->abde",
     "Einsum/Reshape3",
     {{10, 10}, {20, 20}, {30, 30}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 20}, {40, 40}, {50, 50}},
     {{"MatMulV2", 1}, {"Constant", 3}, {"Data", 2}, {"Reshape", 3}}},

    {2,
     "static_shape_scene_2",
     "BTNH,BFNH->BNFT",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{10, 10}, {50, 50}, {30, 30}, {40, 40}},
     {{10, 10}, {30, 30}, {50, 50}, {20, 20}},
     {{"BatchMatMulV2", 1}, {"TransposeD", 2}, {"Data", 2}}},

    {2,
     "static_shape_scene_3",
     "BNFT,BTNH->BFNH",
     "Einsum/Transpose2",
     {{10, 10}, {30, 30}, {50, 50}, {20, 20}},
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{10, 10}, {50, 50}, {30, 30}, {40, 40}},
     {{"BatchMatMulV2", 1}, {"TransposeD", 2}, {"Data", 2}}},

    {2,
     "static_shape_scene_4",
     "abcd,cde->abe",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 20}, {50, 50}},
     {{"BatchMatMulV2", 1}, {"Reshape", 2}, {"Data", 2}, {"Constant", 2}}},

    {2,
     "static_shape_scene_5",
     "abc,cd->abd",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {30, 30}},
     {{30, 30}, {40, 40}},
     {{10, 10}, {20, 20}, {40, 40}},
     {{"BatchMatMulV2", 1}, {"Data", 2}}},

    {2,
     "static_shape_scene_6",
     "abd,cd->abc",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {40, 40}},
     {{30, 30}, {40, 40}},
     {{10, 10}, {20, 20}, {30, 30}},
     {{"BatchMatMulV2", 1}, {"Data", 2}}},

    {2,
     "static_shape_scene_7",
     "abd,abc->cd",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {40, 40}},
     {{10, 10}, {20, 20}, {30, 30}},
     {{30, 30}, {40, 40}},
     {{"MatMulV2", 1}, {"Reshape", 2}, {"Data", 2}, {"Constant", 2}}},

    {2,
     "static_shape_scene_8",
     "abe,cde->abcd",
     "Einsum/Reshape3",
     {{10, 10}, {20, 20}, {50, 50}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{"MatMulV2", 1}, {"Reshape", 3}, {"Data", 2}, {"Constant", 3}}},

    {2,
     "static_shape_scene_9",
     "abe,abcd->cde",
     "Einsum/Reshape3",
     {{10, 10}, {20, 20}, {50, 50}},
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{"MatMulV2", 1}, {"Reshape", 3}, {"Data", 2}, {"Constant", 3}}},

    {2,
     "static_shape_scene_10",
     "BFNH,BTNH->BNFT",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{10, 10}, {50, 50}, {30, 30}, {40, 40}},
     {{10, 10}, {30, 30}, {20, 20}, {50, 50}},
     {{"BatchMatMulV2", 1}, {"Data", 2}, {"TransposeD", 2}}},

    {2,
     "static_shape_scene_11",
     "BFNH,BNFT->BTNH",
     "Einsum/Transpose2",
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{10, 10}, {30, 30}, {20, 20}, {50, 50}},
     {{10, 10}, {50, 50}, {30, 30}, {40, 40}},
     {{"BatchMatMulV2", 1}, {"Data", 2}, {"TransposeD", 2}}},

    {2,
     "static_shape_scene_12",
     "abde,cde->abc",
     "Einsum/BatchMatMul1",
     {{10, 10}, {20, 20}, {40, 40}, {50, 50}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 20}, {30, 30}},
     {{"BatchMatMulV2", 1}, {"Data", 2}, {"Reshape", 2}, {"Constant", 2}}},

    {2,
     "static_shape_scene_13",
     "abde,abc->cde",
     "Einsum/Reshape3",
     {{10, 10}, {20, 20}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 20}, {30, 30}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{"MatMulV2", 1}, {"Data", 2}, {"Reshape", 3}, {"Constant", 3}}},

    {2,
     "static_shape_scene_14",
     "BNFT,BFNH->BTNH",
     "Einsum/Transpose2",
     {{10, 10}, {20, 20}, {30, 30}, {40, 40}},
     {{10, 10}, {30, 30}, {20, 20}, {50, 50}},
     {{10, 10}, {40, 40}, {20, 20}, {50, 50}},
     {{"BatchMatMulV2", 1}, {"Data", 2}, {"TransposeD", 2}}},

    {2,
     "dynamic_shape_scene_1",
     "abc,cde->abde",
     "Einsum/Reshape3",
     {{21, 21}, {31, 31}, {41, 41}},
     {{41, 41}, {52, 52}, {36, 56}},
     {{21, 21}, {31, 31}, {52, 52}, {36, 56}},
     {{"GatherShapes", 1}, {"Reshape", 2}, {"Constant", 1}, {"FlattenV2", 1}, {"Data", 2}, {"MatMulV2", 1}}},

    {2,
     "dynamic_shape_scene_2",
     "BTNH,BFNH->BNFT",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 40}, {30, 30}, {40, 40}},
     {{10, 10}, {50, 50}, {30, 30}, {40, 40}},
     {{10, 10}, {30, 30}, {50, 50}, {20, 40}},
     {{"Transpose", 1}, {"TransposeD", 1}, {"Const", 1}, {"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_3",
     "BNFT,BTNH->BFNH",
     "Einsum/Transpose2",
     {{10, 10}, {20, 40}, {30, 30}, {40, 40}},
     {{10, 10}, {40, 40}, {20, 40}, {50, 50}},
     {{10, 10}, {30, 30}, {20, 40}, {50, 50}},
     {{"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_4",
     "abcd,cde->abe",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 40}, {30, 30}, {40, 40}},
     {{30, 30}, {40, 40}, {50, 50}},
     {{10, 10}, {20, 40}, {50, 50}},
     {{"FlattenV2", 1}, {"Constant", 1}, {"Reshape", 1}, {"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_5",
     "abc,cd->abd",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 40}, {30, 30}},
     {{30, 30}, {30, 60}},
     {{10, 10}, {20, 40}, {30, 60}},
     {{"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_6",
     "abd,cd->abc",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 40}, {30, 30}},
     {{30, 30}, {30, 60}},
     {{10, 10}, {20, 40}, {30, 30}},
     {{"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_7",
     "abd,abc->cd",
     "Einsum/MatMul",
     {{10, 10}, {20, 40}, {30, 30}},
     {{10, 10}, {22, 33}, {40, 40}},
     {{40, 40}, {30, 30}},
     {{"Data", 2}, {"MatMulV2", 1}, {"FlattenV2", 2}}},

    {2,
     "dynamic_shape_scene_8",
     "abe,cde->abcd",
     "Einsum/Reshape3",
     {{39, 39}, {19, 19}, {28, 28}},
     {{38, 38}, {11, 11}, {28, 48}},
     {{39, 39}, {19, 19}, {38, 38}, {11, 11}},
     {{"GatherShapes", 1}, {"Reshape", 2}, {"Constant", 1}, {"FlattenV2", 1}, {"Data", 2}, {"MatMulV2", 1}}},

    {2,
     "dynamic_shape_scene_9",
     "abe,abcd->cde",
     "Einsum/Reshape4",
     {{1, -1}, {14, 34}, {4, 8}},
     {{42, 62}, {7, 14}, {6, 13}, {26, 46}},
     {{6, 13}, {26, 46}, {4, 8}},
     {{"GatherShapes", 1}, {"Reshape", 1}, {"FlattenV2", 3}, {"Data", 2}, {"MatMulV2", 1}}},

    {2,
     "dynamic_shape_scene_10",
     "BFNH,BTNH->BNFT",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 40}, {30, 30}, {40, 40}},
     {{10, 20}, {30, 50}, {23, 43}, {40, 40}},
     {{10, 10}, {30, 30}, {20, 40}, {30, 50}},
     {{"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_11",
     "BFNH,BNFT->BTNH",
     "Einsum/Transpose2",
     {{10, 10}, {20, 20}, {12, 52}, {40, 40}},
     {{10, 10}, {22, 100}, {20, 20}, {50, 50}},
     {{10, 10}, {50, 50}, {22, 52}, {40, 40}},
     {{"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}}},

    {2,
     "dynamic_shape_scene_12",
     "abde,cde->abc",
     "Einsum/BatchMatMul",
     {{10, 10}, {20, 20}, {20, 40}, {30, 50}},
     {{60, 60}, {30, 30}, {50, 50}},
     {{10, 10}, {20, 20}, {60, 60}},
     {{"FlattenV2", 1}, {"Constant", 1}, {"Data", 2}, {"BatchMatMul", 1}, {"Reshape", 1}}},

    {2,
     "dynamic_shape_scene_13",
     "abde,abc->cde",
     "Einsum/Reshape4",
     {{1, -1}, {37, 57}, {8, 28}, {1, -1}},
     {{35, 55}, {57, 77}, {32, 52}},
     {{32, 52}, {8, 28}, {1, -1}},
     {{"GatherShapes", 1}, {"Reshape", 1}, {"FlattenV2", 3}, {"Data", 2}, {"MatMulV2", 1}}},

    {2,
     "dynamic_shape_scene_14",
     "BNFT,BFNH->BTNH",
     "Einsum/Transpose3",
     {{10, 10}, {1, -1}, {20, 20}, {40, 40}},
     {{10, 10}, {20, 20}, {2, 100}, {50, 50}},
     {{10, 10}, {40, 40}, {2, 100}, {50, 50}},
     {{"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}}},

    {1,
     "static_shape_fuzz_single_input_1",
     "nxgb->xb",
     "Einsum/ReduceSum1",
     {{18, 18}, {62, 62}, {55, 55}, {4, 4}},
     {},
     {{62, 62}, {4, 4}},
     {{"TransposeD", 1}, {"ReduceSumD", 1}, {"Data", 1}}},

    {1,
     "static_shape_fuzz_single_input_2",
     "sl->ls",
     "Einsum/Transpose1",
     {{42, 42}, {45, 45}},
     {},
     {{45, 45}, {42, 42}},
     {{"TransposeD", 1}, {"Data", 1}}},

    {2,
     "static_shape_fuzz_two_input_1",
     "nq,n->n",
     "Einsum/Reshape3",
     {{2, 2}, {49, 49}},
     {{2, 2}},
     {{2, 2}},
     {{"ReduceSumD", 1}, {"Data", 2}, {"BatchMatMulV2", 1}, {"Constant", 3}, {"Reshape", 3}}},

    {2,
     "static_shape_fuzz_two_input_2",
     "fhse,rgm->esrfh",
     "Einsum/Transpose2",
     {{2, 2}, {9, 9}, {25, 25}, {32, 32}},
     {{41, 41}, {12, 12}, {45, 45}},
     {{32, 32}, {25, 25}, {41, 41}, {2, 2}, {9, 9}},
     {{"ReduceSumD", 1},
      {"Data", 2},
      {"MatMulV2", 1},
      {"Reshape", 3},
      {"Constant", 3},
      {"TransposeD", 2}}},

    {2,
     "static_shape_fuzz_two_input_3",
     "fytp,hiut->ih",
     "Einsum/Reshape3",
     {{27, 27}, {1, 1}, {63, 63}, {43, 43}},
     {{4, 4}, {13, 13}, {42, 42}, {63, 63}},
     {{13, 13}, {4, 4}},
     {{"ReduceSumD", 2},
      {"Data", 2},
      {"MatMulV2", 1},
      {"Constant", 3},
      {"Reshape", 3},
      {"TransposeD", 2}}},

    {2,
     "static_shape_fuzz_two_input_broad_cast_1",
     "nl,q...ciwd->wq",
     "Einsum/Reshape3",
     {{30, 30}, {41, 41}},
     {{17, 17}, {60, 60}, {13, 13}, {8, 8}, {6, 6}},
     {{8, 8}, {17, 17}},
     {{"ReduceSumD", 2},
      {"Data", 2},
      {"MatMulV2", 1},
      {"Constant", 3},
      {"Reshape", 3},
      {"TransposeD", 1}}},

    {2,
     "static_shape_fuzz_two_input_broad_cast_2",
     "m...x,lzy->m",
     "Einsum/Reshape3",
     {{13, 13}, {64, 64}},
     {{15, 15}, {4, 4}, {33, 33}},
     {{13, 13}},
     {{"ReduceSumD", 2}, {"Data", 2}, {"MatMulV2", 1}, {"Constant", 3}, {"Reshape", 3}}},

    {2,
     "static_shape_fuzz_two_input_broad_cast_3",
     "m...r,...ic->...mci",
     "Einsum/Reshape3",
     {{26, 26}, {25, 25}, {14, 14}},
     {{1, 1}, {51, 51}, {18, 18}},
     {{25, 25}, {26, 26}, {18, 18}, {51, 51}},
     {{"ReduceSumD", 1},
      {"Data", 2},
      {"BatchMatMulV2", 1},
      {"Constant", 3},
      {"Reshape", 3},
      {"TransposeD", 2}}},

};

TEST_P(einsum_fusion_test, general_cases) {
  EinsumTestParam param = GetParam();
  std::cout << "run case " << param.case_name << std::endl;
  ge::Graph graph(param.case_name.c_str());
  ge::op::Data input_x1 = op::Data().set_attr_index(0);
  ge::op::Data input_x2 = op::Data().set_attr_index(1);
  auto einsum = op::Einsum("Einsum");

  auto &input_x1_range = param.input_x1_range;
  auto &input_x2_range = param.input_x2_range;

  std::vector<int64_t> input_x1_vec;
  std::vector<int64_t> input_x2_vec;
  bool dynamic = GetShapeFromRange(input_x1_range, input_x1_vec);
  dynamic = GetShapeFromRange(input_x2_range, input_x2_vec) || dynamic;

  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT16);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x2_desc.SetOriginShape(input_x2_shape);
  if (dynamic) {
    input_x1_desc.SetShapeRange(input_x1_range);
    input_x2_desc.SetShapeRange(input_x2_range);
  }

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(param.input_num);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  if (param.input_num == 2) {
    einsum.set_dynamic_input_x(1, input_x2);
    einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  }

  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation(param.equation);
  einsum.set_attr_N(param.input_num);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  auto &expect_output_range = param.output_range;
  std::vector<int64_t> expect_output_shape;
  GetShapeFromRange(expect_output_range, expect_output_shape);
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  if (dynamic) {
    einsum.GetOutputDesc(0).GetShapeRange(output_range);
    ASSERT_EQ(output_range, expect_output_range);
  }

  std::vector<Operator> inputs{input_x1};
  if (param.input_num == 2) {
    inputs.push_back(input_x2);
  }
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  auto &expected_nodes = param.expected_nodes;
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected_nodes, actual);

  auto last_node = GetNode(param.last_node_name, compute_graph_ptr);
  ASSERT_NE(last_node, nullptr);
  auto op_desc = last_node->GetOpDesc();
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), expect_output_shape);
  if (dynamic) {
    output_range.clear();
    op_desc->MutableOutputDesc(0)->GetShapeRange(output_range);
    ASSERT_EQ(output_range, expect_output_range);
  }
}

INSTANTIATE_TEST_CASE_P(Einsum, einsum_fusion_test, testing::ValuesIn(einsum_general_cases_params));

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_equation_not_match) {
  ge::Graph graph("einsum_fusion_dynamic_test_equation_not_match");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{30, -1};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{30, 30}, {30, 60}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(3, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abd,cd->abc");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, -1, 30};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {20, 40}, {30, 30}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  einsum.set_attr_equation("ab,cd->abc");
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(einsum_fusion_test, einsum_fusion_fuzz_SplitStr2Vector) {
  fe::EinsumPass pass;
  std::string input = "abc,cde,abe->abcd->adce->";
  std::string delimiter = "->";
  std::vector<std::string> output;
  pass.SplitStr2Vector(input, delimiter, output);
  std::vector<std::string> expect_output({"abc,cde,abe", "abcd", "adce"});
  ASSERT_EQ(expect_output, output);

  output.clear();
  delimiter = ",";
  pass.SplitStr2Vector(input, delimiter, output);
  expect_output = {"abc", "cde", "abe->abcd->adce->"};
  ASSERT_EQ(expect_output, output);
}