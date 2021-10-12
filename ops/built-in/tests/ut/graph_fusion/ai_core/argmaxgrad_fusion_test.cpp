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

class argmaxgrad_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "argmaxgrad_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "argmaxgrad_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(argmaxgrad_fusion_pass_test, argmaxgrad_fusion_pass_test_1) {
  ge::Graph graph("argmaxgrad_fusion_pass_test_1");
  auto var_shape = vector<int64_t>({10, 5});
  TensorDesc desc_var(ge::Shape(var_shape), FORMAT_ND, DT_FLOAT16);
  auto var = op::Data("var");
  var.update_input_desc_x(desc_var);
  var.update_output_desc_y(desc_var);

  auto indice_shape = vector<int64_t>({5});
  TensorDesc desc_indice(ge::Shape(indice_shape), FORMAT_ND, DT_INT32);
  auto indice = op::Data("indice");
  indice.update_input_desc_x(desc_indice);
  indice.update_output_desc_y(desc_indice);

  auto update_shape = vector<int64_t>({5});
  TensorDesc desc_update(ge::Shape(update_shape), FORMAT_ND, DT_FLOAT16);
  auto update = op::Data("update");
  update.update_input_desc_x(desc_update);
  update.update_output_desc_y(desc_update);

  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{0};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmaxgrad op
  auto argmaxgrad = op::ArgMaxGrad("ArgMaxGrad");
  argmaxgrad.set_input_var(var)
            .set_input_indices(indice)
            .set_input_updates(update)
            .set_attr_dimension(0);
  std::vector<Operator> inputs{var, indice, update, argmax_multiples};
  std::vector<Operator> outputs{argmaxgrad};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {10, 5};
  ge::DataType expect_datatype = ge::DT_FLOAT16;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxGradD") {
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

TEST_F(argmaxgrad_fusion_pass_test, argmaxgrad_fusion_pass_test_2) {
  ge::Graph graph("argmaxgrad_fusion_pass_test_2");
  auto var_shape = vector<int64_t>({10, 5});
  TensorDesc desc_var(ge::Shape(var_shape), FORMAT_ND, DT_FLOAT16);
  auto var = op::Data("var");
  var.update_input_desc_x(desc_var);
  var.update_output_desc_y(desc_var);

  auto indice_shape = vector<int64_t>({10});
  TensorDesc desc_indice(ge::Shape(indice_shape), FORMAT_ND, DT_INT32);
  auto indice = op::Data("indice");
  indice.update_input_desc_x(desc_indice);
  indice.update_output_desc_y(desc_indice);

  auto update_shape = vector<int64_t>({10});
  TensorDesc desc_update(ge::Shape(update_shape), FORMAT_ND, DT_FLOAT16);
  auto update = op::Data("update");
  update.update_input_desc_x(desc_update);
  update.update_output_desc_y(desc_update);

  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{0};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmaxgrad op
  auto argmaxgrad = op::ArgMaxGrad("ArgMaxGrad");
  argmaxgrad.set_input_var(var)
            .set_input_indices(indice)
            .set_input_updates(update)
            .set_attr_dimension(1);
  std::vector<Operator> inputs{var, indice, update, argmax_multiples};
  std::vector<Operator> outputs{argmaxgrad};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {10, 5};
  ge::DataType expect_datatype = ge::DT_FLOAT16;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxGradD") {
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

TEST_F(argmaxgrad_fusion_pass_test, argmaxgrad_fusion_pass_test_3) {
  ge::Graph graph("argmaxgrad_fusion_pass_test_3");
  auto var_shape = vector<int64_t>({20, 10, 5});
  TensorDesc desc_var(ge::Shape(var_shape), FORMAT_ND, DT_FLOAT16);
  auto var = op::Data("var");
  var.update_input_desc_x(desc_var);
  var.update_output_desc_y(desc_var);

  auto indice_shape = vector<int64_t>({20, 5});
  TensorDesc desc_indice(ge::Shape(indice_shape), FORMAT_ND, DT_INT32);
  auto indice = op::Data("indice");
  indice.update_input_desc_x(desc_indice);
  indice.update_output_desc_y(desc_indice);

  auto update_shape = vector<int64_t>({20, 5});
  TensorDesc desc_update(ge::Shape(update_shape), FORMAT_ND, DT_FLOAT16);
  auto update = op::Data("update");
  update.update_input_desc_x(desc_update);
  update.update_output_desc_y(desc_update);

  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{0};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmaxgrad op
  auto argmaxgrad = op::ArgMaxGrad("ArgMaxGrad");
  argmaxgrad.set_input_var(var)
            .set_input_indices(indice)
            .set_input_updates(update)
            .set_attr_dimension(1);
  std::vector<Operator> inputs{var, indice, update, argmax_multiples};
  std::vector<Operator> outputs{argmaxgrad};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {20, 10, 5};
  ge::DataType expect_datatype = ge::DT_FLOAT16;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxGradD") {
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

TEST_F(argmaxgrad_fusion_pass_test, argmaxgrad_fusion_pass_test_4) {
  ge::Graph graph("argmaxgrad_fusion_pass_test_4");
  auto var_shape = vector<int64_t>({20, 10, 5});
  TensorDesc desc_var(ge::Shape(var_shape), FORMAT_ND, DT_FLOAT16);
  auto var = op::Data("var");
  var.update_input_desc_x(desc_var);
  var.update_output_desc_y(desc_var);

  auto indice_shape = vector<int64_t>({10, 5});
  TensorDesc desc_indice(ge::Shape(indice_shape), FORMAT_ND, DT_INT32);
  auto indice = op::Data("indice");
  indice.update_input_desc_x(desc_indice);
  indice.update_output_desc_y(desc_indice);

  auto update_shape = vector<int64_t>({10, 5});
  TensorDesc desc_update(ge::Shape(update_shape), FORMAT_ND, DT_FLOAT16);
  auto update = op::Data("update");
  update.update_input_desc_x(desc_update);
  update.update_output_desc_y(desc_update);

  auto multiples_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[2]{0};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));

  auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
  // argmaxgrad op
  auto argmaxgrad = op::ArgMaxGrad("ArgMaxGrad");
  argmaxgrad.set_input_var(var)
            .set_input_indices(indice)
            .set_input_updates(update)
            .set_attr_dimension(0);
  std::vector<Operator> inputs{var, indice, update, argmax_multiples};
  std::vector<Operator> outputs{argmaxgrad};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {20, 10, 5};
  ge::DataType expect_datatype = ge::DT_FLOAT16;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxGradD") {
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
