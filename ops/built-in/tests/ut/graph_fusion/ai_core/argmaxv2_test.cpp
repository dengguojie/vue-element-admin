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
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class argmaxv2_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "argmaxv2_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "argmaxv2_fusion_pass_test TearDown" << std::endl;
  }
};

// input dimension is const
TEST_F(argmaxv2_fusion_pass_test, argmaxv2_fusion_pass_test_1) {
  ge::Graph graph("argmaxv2_fusion_pass_test_1");
  auto shape_data = vector<int64_t>({-1, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, 10}, {5, 5}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{0};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmax op
  auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
  argmaxv2.set_input_x(data);
  argmaxv2.set_input_dimension(argmax_multiples);
  std::vector<Operator> inputs{data, argmax_multiples};
  std::vector<Operator> outputs{argmaxv2};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {5};
  ge::DataType expect_datatype = ge::DT_INT32;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxV2") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, true);
  delete[] multiples_tensor_value;
}

TEST_F(argmaxv2_fusion_pass_test, argmaxv2_fusion_pass_test_2) {
  ge::Graph graph("argmaxv2_fusion_pass_test_2");
  auto shape_data = vector<int64_t>({-1, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, -1}, {5, 2147483649}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{1};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmax op
  auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
  argmaxv2.set_input_x(data);
  argmaxv2.set_input_dimension(argmax_multiples);
  std::vector<Operator> inputs{data, argmax_multiples};
  std::vector<Operator> outputs{argmaxv2};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1};
  ge::DataType expect_datatype = ge::DT_INT64;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxV2") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, true);
  delete[] multiples_tensor_value;
}

TEST_F(argmaxv2_fusion_pass_test, argmaxv2_fusion_pass_test_3) {
  ge::Graph graph("argmaxv2_fusion_pass_test_3");
  auto shape_data = vector<int64_t>({2, 32});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{1};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmax op
  auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
  argmaxv2.set_input_x(data);
  argmaxv2.set_input_dimension(argmax_multiples);
  std::vector<Operator> inputs{data, argmax_multiples};
  std::vector<Operator> outputs{argmaxv2};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {2};
  ge::DataType expect_datatype = ge::DT_INT32;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxV2") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, true);
  delete[] multiples_tensor_value;
}
