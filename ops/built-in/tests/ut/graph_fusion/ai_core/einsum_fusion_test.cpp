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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
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
    if (node->GetType() == "BatchMatMulV2") {
      findOp = true;
    }
    if (node->GetType() == "TransposeD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
