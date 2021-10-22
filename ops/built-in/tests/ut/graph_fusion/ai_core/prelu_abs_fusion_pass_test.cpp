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
 * \file prelu_abs_fusion_pass_test.cpp
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

class prelu_abs_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_add SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(prelu_abs_fusion_pass_test, prelu_abs_fusion_pass_test_1) {
  ge::Graph graph("prelu_abs_fusion_pass_test_1");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto abs = op::Abs("abs");
  abs.set_input_x(input0);

  auto sub = op::Sub("sub");
  sub.set_input_x1(input0);
  sub.set_input_x2(abs);

  auto mul = op::Mul("mul");
  mul.set_input_x1(input0);
  mul.set_input_x2(sub);

  auto mul1 = op::Mul("mul1");
  mul1.set_input_x1(input0);
  mul1.set_input_x2(mul);

  auto relu = op::Relu("relu");
  relu.set_input_x(input0);

  auto add = op::Add("add");
  add.set_input_x1(relu);
  add.set_input_x2(mul1);

  auto output0 = op::Data("output0");
  output0.set_input_x(add);

  std::vector<Operator> inputs{input0};
  std::vector<Operator> outputs{output0};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("PReluAbsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool ret1 = false;
  bool ret2 = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PRelu") {
      ret1 = true;
    }
    if (node->GetType() == "Relu") {
      ret2 = false;
    }
  }
  EXPECT_EQ(ret1, true);
  EXPECT_EQ(ret2, false);
}