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

class argmaxwithk_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "argmaxwithk_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "argmaxwithk_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(argmaxwithk_fusion_pass_test, argmaxwithk_fusion_pass_test_1) {
  ge::Graph graph("argmaxwithk_fusion_pass_test_1");
  auto x_shape = vector<int64_t>({10, 5});
  TensorDesc desc_x(ge::Shape(x_shape), FORMAT_ND, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  // argmaxwithk op
  auto argmaxwithk = op::ArgMaxWithK("ArgMaxWithK");
  argmaxwithk.set_input_x(x);
  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{argmaxwithk};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxWithKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {10, 1};
  ge::DataType expect_datatype = ge::DT_INT32;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxWithKD") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(argmaxwithk_fusion_pass_test, argmaxwithk_fusion_pass_test_2) {
  ge::Graph graph("argmaxwithk_fusion_pass_test_2");
  auto x_shape = vector<int64_t>({10, 5});
  TensorDesc desc_x(ge::Shape(x_shape), FORMAT_ND, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto topk_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(topk_shape, FORMAT_ND, DT_INT32);
  Tensor topk_tensor(desc_input_size_1);
  uint32_t *topk_tensor_value = new uint32_t[2]{0};
  topk_tensor.SetData((uint8_t *) topk_tensor_value, sizeof(uint32_t));

  auto argmax_topk = op::Constant("topk").set_attr_value(topk_tensor);
  // argmaxwithk op
  auto argmaxwithk = op::ArgMaxWithK("ArgMaxWithK");
  argmaxwithk.set_input_x(x)
             .set_attr_topk(0);
  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{argmaxwithk};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxWithKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {10, 1};
  ge::DataType expect_datatype = ge::DT_INT32;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxWithKD") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, false);
  delete[] topk_tensor_value;
}

TEST_F(argmaxwithk_fusion_pass_test, argmaxwithk_fusion_pass_test_3) {
  ge::Graph graph("argmaxwithk_fusion_pass_test_3");
  auto x_shape = vector<int64_t>({10, 5});
  TensorDesc desc_x(ge::Shape(x_shape), FORMAT_ND, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto topk_shape = ge::Shape({1});
  TensorDesc desc_input_size_1(topk_shape, FORMAT_ND, DT_INT32);
  Tensor topk_tensor(desc_input_size_1);
  uint32_t *topk_tensor_value = new uint32_t[2]{0};
  topk_tensor.SetData((uint8_t *) topk_tensor_value, sizeof(uint32_t));

  auto argmax_topk = op::Constant("topk").set_attr_value(topk_tensor);
  // argmaxwithk op
  auto argmaxwithk = op::ArgMaxWithK("ArgMaxWithK");
  argmaxwithk.set_input_x(x)
             .set_attr_topk(2);
  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{argmaxwithk};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ArgMaxWithKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  std::vector<int64_t> expected_shape = {10, 1};
  ge::DataType expect_datatype = ge::DT_INT32;

  bool findD = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ArgMaxWithKD") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      auto out_datatype = output_desc.GetDataType();
      EXPECT_EQ(dims, expected_shape);
      EXPECT_EQ(out_datatype, expect_datatype);
    }
  }
  EXPECT_EQ(findD, false);
  delete[] topk_tensor_value;
}
