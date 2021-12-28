#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "correlation.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class correlation_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "correlation_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "correlation_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_1) {
  ge::Graph graph("correlation_fusion_pass_test_1");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_2) {
  ge::Graph graph("correlation_fusion_pass_test_2");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 32, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(32);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_3) {
  ge::Graph graph("correlation_fusion_pass_test_3");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 25, 25, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_4) {
  ge::Graph graph("correlation_fusion_pass_test_4");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 25, 25, 32};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(32);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_5) {
  ge::Graph graph("correlation_fusion_pass_test_5");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{16, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_6) {
  ge::Graph graph("correlation_fusion_pass_test_6");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_7) {
  ge::Graph graph("correlation_fusion_pass_test_7");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 32, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_ND, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_8) {
  ge::Graph graph("correlation_fusion_pass_test_8");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{32, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{32, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{32, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_9) {
  ge::Graph graph("correlation_fusion_pass_test_9");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{32, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    
  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{32, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{32, 32, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(32);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_10) {
  ge::Graph graph("correlation_fusion_pass_test_10");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{16, 32, 5, 5};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_HWCN, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 32, 29, 29};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NCHW, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 1, 25, 25};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_11) {
  ge::Graph graph("correlation_fusion_pass_test_11");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 25, 25, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_12) {
  ge::Graph graph("correlation_fusion_pass_test_12");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{1, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 25, 25, 32};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);

  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(32);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_13) {
  ge::Graph graph("correlation_fusion_pass_test_13");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{16, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{1, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{1, 25, 25, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_14) {
  ge::Graph graph("correlation_fusion_pass_test_14");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{32, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{32, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{32, 25, 25, 1};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(1);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_15) {
  ge::Graph graph("correlation_fusion_pass_test_15");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{32, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{32, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{32, 25, 25, 32};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(32);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}

TEST_F(correlation_fusion_pass_test, correlation_fusion_pass_test_16) {
  ge::Graph graph("correlation_fusion_pass_test_16");

  auto x1 = op::Data("filter");
  std::vector<int64_t> dims_x1{32, 5, 5, 32};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensor_desc_x1(shape_x1, FORMAT_NHWC, DT_FLOAT16);

  auto x2 = op::Data("x");
  std::vector<int64_t> dims_x2{32, 29, 29, 32};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensor_desc_x2(shape_x2, FORMAT_NHWC, DT_FLOAT16);

  std::vector<int64_t> dims_y{32, 25, 25, 32};
  ge::Shape shape_y(dims_y);
  ge::TensorDesc tensor_desc_y(shape_y, FORMAT_NHWC, DT_FLOAT16);
  x1.update_input_desc_x(tensor_desc_x1);
  x1.update_output_desc_y(tensor_desc_x1);
  x2.update_input_desc_x(tensor_desc_x2);
  x2.update_output_desc_y(tensor_desc_x2);
  auto correlation_op = op::Correlation("Correlation");
  correlation_op.set_input_filter(x1)
                .set_input_x(x2)
                .set_attr_groups(16);
  correlation_op.update_input_desc_filter(tensor_desc_x1);
  correlation_op.update_input_desc_x(tensor_desc_x2);
  correlation_op.update_output_desc_y(tensor_desc_y);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{correlation_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CorrelationFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findC = true;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Correlation") {
            findC = true;
        }
    }
  EXPECT_EQ(findC, true);
}
