#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class topk_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_add SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(topk_fusion_test, sub_fusion_test_1) {
  ge::Graph graph("sub_fusion_test_1");
  // set input data 1
  auto sub_input_data = op::Data("sub_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_INT32);
  sub_input_data.update_input_desc_x(tensorDesc);
  sub_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto sub_input_data2 = op::Data("sub_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_INT32);
  sub_input_data2.update_input_desc_x(tensorDesc2);
  sub_input_data2.update_output_desc_y(tensorDesc2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(sub_input_data).set_input_x2(sub_input_data2);
  // set graph and res
  std::vector<Operator> inputs{sub_input_data, sub_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findSub = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Sub") {
      findSub = true;
      break;
    }
  }
  EXPECT_EQ(findSub, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_2) {
  ge::Graph graph("sub_fusion_test_2");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_INT32);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_INT32);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_3) {
  ge::Graph graph("sub_fusion_test_3");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_FLOAT16);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_4) {
  ge::Graph graph("sub_fusion_test_4");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_FLOAT);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_5) {
  ge::Graph graph("sub_fusion_test_5");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_UINT8);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_UINT8);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_6) {
  ge::Graph graph("sub_fusion_test_6");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_INT64);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_INT64);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_7) {
  ge::Graph graph("sub_fusion_test_7");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_INT16);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_INT16);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}

TEST_F(topk_fusion_test, sub_fusion_test_8) {
  ge::Graph graph("sub_fusion_test_8");
  // set input data 1
  auto mul_input_data = op::Data("mul_input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_UINT16);
  mul_input_data.update_input_desc_x(tensorDesc);
  mul_input_data.update_output_desc_y(tensorDesc);

  // set input data 2
  auto mul_input_data2 = op::Data("mul_input_data2");
  std::vector<int64_t> dims2{3, 32};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, FORMAT_ND, DT_UINT16);
  mul_input_data2.update_input_desc_x(tensorDesc2);
  mul_input_data2.update_output_desc_y(tensorDesc2);

  // set mul as input
  auto mul_op = op::Mul("mul_0").set_input_x1(mul_input_data).set_input_x2(mul_input_data2);

  // set fusion
  auto sub_op = op::Sub("sub_0").set_input_x1(mul_op).set_input_x2(mul_op);
  // set graph and res
  std::vector<Operator> inputs{mul_input_data, mul_input_data2};
  std::vector<Operator> outputs{sub_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  // common func
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SubFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      findMul = true;
      break;
    }
  }
  EXPECT_EQ(findMul, true);
}