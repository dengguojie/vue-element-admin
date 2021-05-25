#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "globalavgpool.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class Globalavgpool__fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Globalavgpool__fusion_test SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Globalavgpool__fusion_test TearDown" << std::endl;
  }
};

TEST_F(Globalavgpool__fusion_test, globalavgpool_fusion_test_1) {
  ge::Graph graph("globalavgpool_fusion_test_1");
  auto globalavgpool_input_data = op::Data("globalavgpool_input_data");

  std::vector<int64_t> dims_x{1, 5, 5, 4, 3};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensordescX(shape_x, ge::FORMAT_NCDHW, ge::DT_FLOAT16);
  globalavgpool_input_data.update_input_desc_x(tensordescX);

  std::vector<int64_t> dims_y{1, 5, 1, 1, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensordescY(shape_y, ge::FORMAT_NCDHW, ge::DT_FLOAT16);
  globalavgpool_input_data.update_output_desc_y(tensordescY);

  auto op = op::GlobalAveragePool("globalavgpool");
  op.set_input_x(globalavgpool_input_data);

  std::vector<Operator> inputs{globalavgpool_input_data};
  std::vector<Operator> outputs{op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("Globalavgpoolpass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool match = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ReduceMeanD") {
      match = true;
    }
  }
  EXPECT_EQ(match, true);
}

TEST_F(Globalavgpool__fusion_test, globalavgpool_fusion_test_2) {
  ge::Graph graph("globalavgpool_fusion_test_2");
  auto globalavgpool_input_data = op::Data("globalavgpool_input_data");

  std::vector<int64_t> dims_x{1, 3, 5, 5};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensordescX(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  globalavgpool_input_data.update_input_desc_x(tensordescX);

  std::vector<int64_t> dims_y{1, 3, 1, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensordescY(shape_y, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  globalavgpool_input_data.update_output_desc_y(tensordescY);

  auto op = op::GlobalAveragePool("GlobalAveragePool");
  op.set_input_x(globalavgpool_input_data);
  std::vector<Operator> inputs{globalavgpool_input_data};
  std::vector<Operator> outputs{op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("Globalavgpoolpass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool match = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ReduceMeanD") {
      match = true;
    }
  }
  EXPECT_EQ(match, true);
}

TEST_F(Globalavgpool__fusion_test, globalavgpool_fusion_test_3) {
  ge::Graph graph("globalavgpool_fusion_test_3");
  auto globalavgpool_input_data = op::Data("globalavgpool_input_data");

  std::vector<int64_t> dims_x{1, 5, 5};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensordescX(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  globalavgpool_input_data.update_input_desc_x(tensordescX);

  std::vector<int64_t> dims_y{1, 5, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensordescY(shape_y, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  globalavgpool_input_data.update_output_desc_y(tensordescY);

  auto op = op::GlobalAveragePool("GlobalAveragePool");
  op.set_input_x(globalavgpool_input_data);
  std::vector<Operator> inputs{globalavgpool_input_data};
  std::vector<Operator> outputs{op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("Globalavgpoolpass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool match = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ReduceMeanD") {
      match = true;
    }
  }
  EXPECT_EQ(match, true);
}

TEST_F(Globalavgpool__fusion_test, globalavgpool_fusion_test_4) {
  ge::Graph graph("globalavgpool_fusion_test_4");
  auto globalavgpool_input_data = op::Data("globalavgpool_input_data");

  std::vector<int64_t> dims{1, 5};
  ge::Shape shape(dims);
  ge::TensorDesc tensordesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  globalavgpool_input_data.update_input_desc_x(tensordesc);
  globalavgpool_input_data.update_output_desc_y(tensordesc);

  auto op = op::GlobalAveragePool("GlobalAveragePool");
  op.set_input_x(globalavgpool_input_data);
  std::vector<Operator> inputs{globalavgpool_input_data};
  std::vector<Operator> outputs{op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("Globalavgpoolpass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool match = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ReduceMeanD") {
      match = true;
    }
  }
  EXPECT_EQ(match, true);
}