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

/*!
 * \file range_fusion_pass_test.cpp
 * \brief
 */
#include "nn_batch_norm_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class range_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "range_fusion_pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "range_fusion_pass TearDown" << std::endl;
  }
};

TEST_F(range_fusion_pass_test, range_fusion_pass_test_1) {
  ge::Graph graph("range_fusion_pass_test");

  auto input0 = op::Const("input0");
  Tensor axis;
  float* dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  axis.SetData((uint8_t*)dataValue, 4);
  TensorDesc desc_data(ge::Shape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  input0.set_attr_value(axis);
  input0.update_output_desc_y(desc_data);
  delete[] dataValue;

  auto input1 = op::Const("input1");
  input1.set_attr_value(axis);
  input1.update_output_desc_y(desc_data);
  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(desc_data);

  auto range = op::Range("range");
  range.set_input_start(input0);
  range.set_input_limit(input1);
  range.set_input_delta(input2);

  auto output0 = op::Data("output0");
  output0.set_input_x(range);

  std::vector<Operator> inputs{input0, input1, input2};
  std::vector<Operator> outputs{output0};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("RangeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool ret = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RangeD") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, true);
}

TEST_F(range_fusion_pass_test, range_fusion_pass_test_2) {
  ge::Graph graph("range_fusion_pass_test_2");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms, ge::FORMAT_NHWC, ge::DT_FLOAT);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto input1 = op::Data("input1");
  input1.update_input_desc_x(tensorDescMs);
  input1.update_output_desc_y(tensorDescMs);

  auto input2 = op::Data("input2");
  input2.update_input_desc_x(tensorDescMs);
  input2.update_output_desc_y(tensorDescMs);

  auto range = op::Range("range");
  range.set_input_start(input0);
  range.set_input_limit(input1);
  range.set_input_delta(input2);

  auto output0 = op::Data("output0");
  output0.set_input_x(range);

  std::vector<Operator> inputs{input0, input1, input2};
  std::vector<Operator> outputs{output0};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("RangeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool ret = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RangeD") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, false);
}

TEST_F(range_fusion_pass_test, range_fusion_pass_test_3) {
  ge::Graph graph("range_fusion_pass_test_3");

  auto input0 = op::Const("input0");
  Tensor axis;
  float* dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_INT32));
  axis.SetData((uint8_t*)dataValue, 4);
  TensorDesc desc_data(ge::Shape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  input0.set_attr_value(axis);
  input0.update_output_desc_y(desc_data);
  delete[] dataValue;

  auto input1 = op::Const("input1");
  input1.set_attr_value(axis);
  input1.update_output_desc_y(desc_data);
  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(desc_data);

  auto range = op::Range("range");
  range.set_input_start(input0);
  range.set_input_limit(input1);
  range.set_input_delta(input2);

  auto output0 = op::Data("output0");
  output0.set_input_x(range);

  std::vector<Operator> inputs{input0, input1, input2};
  std::vector<Operator> outputs{output0};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("RangeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool ret = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RangeD") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, true);
}