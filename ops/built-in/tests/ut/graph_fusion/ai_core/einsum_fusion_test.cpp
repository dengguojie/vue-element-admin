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
#include "array_ops.h"
#include "matrix_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class einsum_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "einsum_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "einsum_fusion_test TearDown" << std::endl;
  }
};

TEST_F(einsum_fusion_test, einsum_fusion_test_1) {
  ge::Graph graph("einsum_fusion_test_1");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 40, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abc,cde->abde");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_2) {
  ge::Graph graph("einsum_fusion_test_2");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 50, 30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 30, 50, 20};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BTNH,BFNH->BNFT");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_3) {
  ge::Graph graph("einsum_fusion_test_3");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 30, 50, 20};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 20, 30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 50, 30, 20};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BNFT,BTNH->BFNH");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_4) {
  ge::Graph graph("einsum_fusion_test_4");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abcd,cde->abe");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_5) {
  ge::Graph graph("einsum_fusion_test_5");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 40};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abc,cd->abd");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_6) {
  ge::Graph graph("einsum_fusion_test_6");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 30};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abd,cd->abc");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_7) {
  ge::Graph graph("einsum_fusion_test_7");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 20, 30};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{30, 40};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abd,abc->cd");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_8) {
  ge::Graph graph("einsum_fusion_test_8");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 50};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 30, 40};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abe,cde->abcd");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_9) {
  ge::Graph graph("einsum_fusion_test_9");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 50};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 20, 30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{30, 40, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abe,abcd->cde");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_10) {
  ge::Graph graph("einsum_fusion_test_10");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 50, 30, 40};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 30, 20, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BFNH,BTNH->BNFT");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_11) {
  ge::Graph graph("einsum_fusion_test_11");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 30, 20, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 50, 30, 40};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BFNH,BNFT->BTNH");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_12) {
  ge::Graph graph("einsum_fusion_test_12");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 40, 50};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{30, 40, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 20, 30};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abde,cde->abc");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_13) {
  ge::Graph graph("einsum_fusion_test_13");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 40, 50};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 20, 30};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{30, 40, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abde,abc->cde");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(einsum_fusion_test, einsum_fusion_test_14) {
  ge::Graph graph("einsum_fusion_test_14");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, 30, 40};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> input_x2_vec{10, 30, 20, 50};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  std::vector<int64_t> output_vec{10, 40, 20, 50};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BNFT,BFNH->BTNH");
  einsum.set_attr_N(2);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_1) {
  ge::Graph graph("einsum_fusion_dynamic_test_1");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{21, 31, 41};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{21, 21}, {31, 31}, {41, 41}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{41, 52, -1};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{41, 41}, {52, 52}, {36, 56}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abc,cde->abde");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), ge::GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{21, 31, 52, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{21, 21}, {31, 31}, {52, 52}, {36, 56}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"GatherShapes", 1}, {"Reshape", 2}, {"Constant", 1}, {"FlattenV2", 1}, {"Data", 2}, {"MatMulV2", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_2) {
  ge::Graph graph("einsum_fusion_dynamic_test_2");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{10, 50, 30, 40};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 10}, {50, 50}, {30, 30}, {40, 40}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BTNH,BFNH->BNFT");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 30, 50, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {30, 30}, {50, 50}, {20, 40}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Transpose", 1}, {"TransposeD", 1}, {"Const", 1}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_3) {
  ge::Graph graph("einsum_fusion_dynamic_test_3");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{10, 40, -1, 50};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 10}, {40, 40}, {20, 40}, {50, 50}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BNFT,BTNH->BFNH");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 30, -1, 50};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {30, 30}, {20, 40}, {50, 50}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_4) {
  ge::Graph graph("einsum_fusion_dynamic_test_4");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{30, 40, 50};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{30, 30}, {40, 40}, {50, 50}};
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
  einsum.set_attr_equation("abcd,cde->abe");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, -1, 50};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {20, 40}, {50, 50}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"FlattenV2", 1}, {"Constant", 1}, {"Reshape", 1}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_5) {
  ge::Graph graph("einsum_fusion_dynamic_test_5");
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
  einsum.set_attr_equation("abc,cd->abd");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, -1, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {20, 40}, {30, 60}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_6) {
  ge::Graph graph("einsum_fusion_dynamic_test_6");
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
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_7) {
  ge::Graph graph("einsum_fusion_dynamic_test_7");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{10, -1, 40};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 10}, {22, 33}, {40, 40}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(2, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abd,abc->cd");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{40, 30};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{40, 40}, {30, 30}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Data", 2}, {"MatMulV2", 1}, {"FlattenV2", 2}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_8) {
  ge::Graph graph("einsum_fusion_dynamic_test_8");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{39, 19, 28};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{39, 39}, {19, 19}, {28, 28}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{38, 11, -1};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{38, 38}, {11, 11}, {28, 48}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(2, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("abe,cde->abcd");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), ge::GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{39, 19, 38, 11};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{39, 39}, {19, 19}, {38, 38}, {11, 11}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"GatherShapes", 1}, {"Reshape", 2}, {"Constant", 1}, {"FlattenV2", 1}, {"Data", 2}, {"MatMulV2", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_9) {
  ge::Graph graph("einsum_fusion_dynamic_test_9");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{-1, -1, -1};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{1, -1}, {14, 34}, {4, 8}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{-1, -1, -1, -1};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{42, 62}, {7, 14}, {6, 13}, {26, 46}};
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
  einsum.set_attr_equation("abe,abcd->cde");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), ge::GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{-1, -1, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{6, 13}, {26, 46}, {4, 8}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"GatherShapes", 1}, {"Reshape", 1}, {"FlattenV2", 3}, {"Data", 2}, {"MatMulV2", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_10) {
  ge::Graph graph("einsum_fusion_dynamic_test_10");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 30, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 40}, {30, 30}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{-1, -1, -1, 40};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 20}, {30, 50}, {23, 43}, {40, 40}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BFNH,BTNH->BNFT");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 30, -1, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {30, 30}, {20, 40}, {30, 50}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_11) {
  ge::Graph graph("einsum_fusion_dynamic_test_11");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, -1, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 20}, {12, 52}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{10, -1, 20, 50};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 10}, {22, 100}, {20, 20}, {50, 50}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BFNH,BNFT->BTNH");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 50, -1, 40};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {50, 50}, {22, 52}, {40, 40}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_12) {
  ge::Graph graph("einsum_fusion_dynamic_test_12");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, 20, -1, -1};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {20, 20}, {20, 40}, {30, 50}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{60, 30, 50};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{60, 60}, {30, 30}, {50, 50}};
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
  einsum.set_attr_equation("abde,cde->abc");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 20, 60};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {20, 20}, {60, 60}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"FlattenV2", 1}, {"Constant", 1}, {"Data", 2}, {"BatchMatMul", 1}, {"Reshape", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_13) {
  ge::Graph graph("einsum_fusion_dynamic_test_13");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{-1, -1, -1, -1};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{1, -1}, {37, 57}, {8, 28}, {1, -1}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{-1, -1, -1};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{35, 55}, {57, 77}, {32, 52}};
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
  einsum.set_attr_equation("abde,abc->cde");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), ge::GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{-1, -1, -1};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{32, 52}, {8, 28}, {1, -1}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"GatherShapes", 1}, {"Reshape", 1}, {"FlattenV2", 3}, {"Data", 2}, {"MatMulV2", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

TEST_F(einsum_fusion_test, einsum_fusion_dynamic_test_14) {
  ge::Graph graph("einsum_fusion_dynamic_test_14");
  auto input_x1 = op::Data().set_attr_index(0);
  auto input_x2 = op::Data().set_attr_index(0);
  auto einsum = op::Einsum("einsum");

  std::vector<int64_t> input_x1_vec{10, -1, 20, 40};
  vector<std::pair<int64_t, int64_t>> input_x1_range = {{10, 10}, {1, -1}, {20, 20}, {40, 40}};
  ge::Shape input_x1_shape(input_x1_vec);
  ge::TensorDesc input_x1_desc(input_x1_shape, FORMAT_ND, DT_FLOAT);
  input_x1_desc.SetOriginShape(input_x1_shape);
  input_x1_desc.SetShapeRange(input_x1_range);

  std::vector<int64_t> input_x2_vec{10, 20, -1, 50};
  vector<std::pair<int64_t, int64_t>> input_x2_range = {{10, 10}, {20, 20}, {2, 100}, {50, 50}};
  ge::Shape input_x2_shape(input_x2_vec);
  ge::TensorDesc input_x2_desc(input_x2_shape, FORMAT_ND, DT_FLOAT);
  input_x2_desc.SetOriginShape(input_x2_shape);
  input_x2_desc.SetShapeRange(input_x2_range);

  ge::Shape output_shape(std::vector<int64_t>(4, -1));
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  output_desc.SetOriginShape(output_shape);

  einsum.create_dynamic_input_x(2);
  einsum.set_dynamic_input_x(0, input_x1);
  einsum.set_dynamic_input_x(1, input_x2);
  einsum.update_dynamic_input_desc_x(0, input_x1_desc);
  einsum.update_dynamic_input_desc_x(1, input_x2_desc);
  einsum.update_output_desc_y(output_desc);
  einsum.set_attr_equation("BNFT,BFNH->BTNH");
  einsum.set_attr_N(2);

  ASSERT_EQ(einsum.InferShapeAndType(), GRAPH_SUCCESS);
  std::vector<int64_t> expect_output_shape{10, 40, -1, 50};
  ASSERT_EQ(einsum.GetOutputDesc(0).GetShape().GetDims(), expect_output_shape);
  vector<std::pair<int64_t, int64_t>> output_range;
  einsum.GetOutputDesc(0).GetShapeRange(output_range);
  vector<std::pair<int64_t, int64_t>> expect_output_range = {{10, 10}, {40, 40}, {2, 100}, {50, 50}};
  ASSERT_EQ(output_range, expect_output_range);

  std::vector<Operator> inputs{input_x1, input_x2};
  std::vector<Operator> outputs{einsum};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("EinsumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  ASSERT_EQ(ret, SUCCESS);

  std::map<std::string, uint32_t> expected = {
      {"Transpose", 2}, {"Const", 2}, {"Data", 2}, {"BatchMatMul", 1}};
  std::map<std::string, uint32_t> actual;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    actual[node->GetType()]++;
  }
  ASSERT_EQ(expected, actual);
}

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
