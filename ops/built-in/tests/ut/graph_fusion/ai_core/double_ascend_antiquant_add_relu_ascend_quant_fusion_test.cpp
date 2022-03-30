/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
#include <algorithm>

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"

#include "elewise_calculation_ops.h" // for Add op
#include "nonlinear_fuc_ops.h" // for Relu op
#include "quantize_ops.h" // for quant op
#include "array_ops.h"  // for Data op

using namespace ge;
using namespace op;

class DoubleAscendAntiQuantAddReluAscendQuantFusionTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DoubleAscendAntiQuantAddReluAscendQuantFusionTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DoubleAscendAntiQuantAddReluAscendQuantFusionTest TearDown" << std::endl;
  }
};

static ge::TensorDesc CreateTensor(const DataType& dtype,
                                   const ge::Shape& shape, const ge::Shape& originShape,
                                   const ge::Format& format, const ge::Format& originFormat) {
  TensorDesc tensorDesc(shape, format, dtype);
  tensorDesc.SetOriginShape(originShape);
  tensorDesc.SetOriginFormat(originFormat);
  return tensorDesc;
}

static op::Data CreateDataNode(const char* name, const DataType& dtype,
                               const std::vector<int64_t>& dims, const std::vector<int64_t>& originDims,
                               const ge::Format& format, const ge::Format& originFormat) {
  auto data = op::Data(name);
  TensorDesc tensorInput = CreateTensor(dtype, ge::Shape(dims), ge::Shape(originDims), format, originFormat);
  data.update_input_desc_x(tensorInput);
  data.update_output_desc_y(tensorInput);
  return data;
}

static op::AscendAntiQuant CreateAscendAntiQuantNode(const char* name, Operator& input_x,
                                                     float scale, float offset,
                                                     bool sqrt_mode = false) {
  return op::AscendAntiQuant(name).set_input_x(input_x)
                                  .set_attr_scale(scale)
                                  .set_attr_offset(offset)
                                  .set_attr_sqrt_mode(sqrt_mode);
}

static op::AscendQuant CreateAscendQuantNode(const char* name, Operator& input_x,
                                             float scale, float offset,
                                             bool sqrt_mode = false, const char* round_mode = "Round",
                                             int dst_type = DT_INT8) {
  return op::AscendQuant(name).set_input_x(input_x)
                              .set_attr_scale(scale)
                              .set_attr_offset(offset)
                              .set_attr_sqrt_mode(sqrt_mode)
                              .set_attr_round_mode(round_mode)
                              .set_attr_dst_type(dst_type);
}

// basic funcion
TEST_F(DoubleAscendAntiQuantAddReluAscendQuantFusionTest, basic_func) {
  // input to AscendAntiQuant
  auto input1 = CreateDataNode("input_1", DT_INT8,
                               {16, 6, 79, 69, 32}, {16, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);
  auto input2 = CreateDataNode("input_2", DT_INT8,
                               {16, 6, 79, 69, 32}, {16, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);

  auto ascendAntiQuant1 = CreateAscendAntiQuantNode("ascend_anti_quant_1", input1, 0.1, 7);
  auto ascendAntiQuant2 = CreateAscendAntiQuantNode("ascend_anti_quant_2", input2, 0.2, 17, true);
  auto add = op::Add("add").set_input_x1(ascendAntiQuant1).set_input_x2(ascendAntiQuant2);
  auto relu = op::Relu("relu").set_input_x(add);
  auto ascendQuant = CreateAscendQuantNode("ascend_quant", relu, 1.1, 107, false, "Ceil", DT_INT8);

  // graph
  std::vector<Operator> inputs{input1, input2};
  std::vector<Operator> outputs{ascendQuant};
  ge::Graph graph("DoubleAscendAntiQuantAddReluAscendQuantFusionTest_basic_func");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DoubleAscendAntiQuantAddReluAscendQuantFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  std::unordered_map<string, std::pair<int, int>> checkMap;
  checkMap["Relu"] = std::pair<int, int>(0, 1);
  checkMap["Add"] = std::pair<int, int>(0, 1);
  checkMap["Cast"] = std::pair<int, int>(0, 3);
  checkMap["Adds"] = std::pair<int, int>(0, 3);
  checkMap["Muls"] = std::pair<int, int>(0, 4);
  checkMap["Ceil"] = std::pair<int, int>(0, 1);

  std::unordered_map<string, std::pair<int, int>>::iterator iter;
  for (const auto node: compute_graph_ptr->GetAllNodes()) {
      iter = checkMap.find(node->GetType());
      if (iter != checkMap.end()) {
        iter->second.first++;
      }
  }

  bool result = std::all_of(checkMap.cbegin(), checkMap.cend(),
                            [](const auto &it) { return it.second.first == it.second.second; });
  EXPECT_EQ(result, true);
}

// basic funcion
TEST_F(DoubleAscendAntiQuantAddReluAscendQuantFusionTest, basic_func2) {
  // input to AscendAntiQuant
  auto input1 = CreateDataNode("input_1", DT_INT8,
                               {16, 6, 79, 69, 32}, {16, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);
  auto input2 = CreateDataNode("input_2", DT_INT8,
                               {16, 6, 79, 69, 32}, {16, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);

  auto ascendAntiQuant1 = CreateAscendAntiQuantNode("ascend_anti_quant_1", input1, 0.1, 7, false);
  auto ascendAntiQuant2 = CreateAscendAntiQuantNode("ascend_anti_quant_2", input2, 0.2, 17, true);
  auto add = op::Add("add").set_input_x1(ascendAntiQuant1).set_input_x2(ascendAntiQuant2);
  auto relu = op::Relu("relu").set_input_x(add);
  auto ascendQuant = CreateAscendQuantNode("ascend_quant", relu, 1.1, 107, true, "Trunc", DT_INT8);

  // graph
  std::vector<Operator> inputs{input1, input2};
  std::vector<Operator> outputs{ascendQuant};
  ge::Graph graph("DoubleAscendAntiQuantAddReluAscendQuantFusionTest_basic_func2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DoubleAscendAntiQuantAddReluAscendQuantFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  std::unordered_map<string, std::pair<int, int>> checkMap;
  checkMap["Relu"] = std::pair<int, int>(0, 1);
  checkMap["Add"] = std::pair<int, int>(0, 1);
  checkMap["Cast"] = std::pair<int, int>(0, 3);
  checkMap["Adds"] = std::pair<int, int>(0, 3);
  checkMap["Muls"] = std::pair<int, int>(0, 5);
  checkMap["Round"] = std::pair<int, int>(0, 0);

  std::unordered_map<string, std::pair<int, int>>::iterator iter;
  for (const auto node: compute_graph_ptr->GetAllNodes()) {
      iter = checkMap.find(node->GetType());
      if (iter != checkMap.end()) {
        iter->second.first++;
      }
  }

  bool result = std::all_of(checkMap.cbegin(), checkMap.cend(),
                            [](const auto &it) { return it.second.first == it.second.second; });
  EXPECT_EQ(result, true);
}

// basic funcion
TEST_F(DoubleAscendAntiQuantAddReluAscendQuantFusionTest, dynamic_should_not_fusion) {
  // input to AscendAntiQuant
  auto input1 = CreateDataNode("input_1", DT_INT8,
                               {-1, 6, 79, 69, 32}, {-1, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);
  auto input2 = CreateDataNode("input_2", DT_INT8,
                               {-1, 6, 79, 69, 32}, {-1, 79, 69, 512},
                               FORMAT_NC1HWC0, FORMAT_NHWC);

  auto ascendAntiQuant1 = CreateAscendAntiQuantNode("ascend_anti_quant_1", input1, 0.1, 7, false);
  auto ascendAntiQuant2 = CreateAscendAntiQuantNode("ascend_anti_quant_2", input2, 0.2, 17, true);
  auto add = op::Add("add").set_input_x1(ascendAntiQuant1).set_input_x2(ascendAntiQuant2);
  auto relu = op::Relu("relu").set_input_x(add);
  auto ascendQuant = CreateAscendQuantNode("ascend_quant", relu, 1.1, 107, true, "Trunc", DT_INT8);

  // graph
  std::vector<Operator> inputs{input1, input2};
  std::vector<Operator> outputs{ascendQuant};
  ge::Graph graph("DoubleAscendAntiQuantAddReluAscendQuantFusionTest_basic_func2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DoubleAscendAntiQuantAddReluAscendQuantFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  std::unordered_map<string, std::pair<int, int>> checkMap;
  checkMap["Relu"] = std::pair<int, int>(0, 1);
  checkMap["Add"] = std::pair<int, int>(0, 1);
  checkMap["Cast"] = std::pair<int, int>(0, 0);
  checkMap["Adds"] = std::pair<int, int>(0, 0);
  checkMap["Muls"] = std::pair<int, int>(0, 0);
  checkMap["Round"] = std::pair<int, int>(0, 0);
  checkMap["AscendAntiQuant"] = std::pair<int, int>(0, 2);
  checkMap["AscendQuant"] = std::pair<int, int>(0, 1);

  std::unordered_map<string, std::pair<int, int>>::iterator iter;
  for (const auto node: compute_graph_ptr->GetAllNodes()) {
      iter = checkMap.find(node->GetType());
      if (iter != checkMap.end()) {
        iter->second.first++;
      }
  }

  bool result = std::all_of(checkMap.cbegin(), checkMap.cend(),
                            [](const auto &it) { return it.second.first == it.second.second; });
  EXPECT_EQ(result, true);
}
