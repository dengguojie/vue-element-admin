#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class diag_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(diag_fusion_test, diag_fusion_test_1) {
    ge::Graph graph("diag_fusion_test_1");
    auto diag_input_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);
    diag_input_data.update_input_desc_x(tensorDesc);
    diag_input_data.update_output_desc_y(tensorDesc);
    auto diag_op = op::Diag("diag_0");
    diag_op.set_input_x(diag_input_data);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(diag_op);
    std::vector<Operator> inputs{diag_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("DiagFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
    bool findDiagD = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{3, 32, 3, 32};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DiagD") {
            findDiagD = true;
            auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
            std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findDiagD, true);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(diag_fusion_test, diag_fusion_test_2) {
    ge::Graph graph("diag_fusion_test_2");
    auto diag_input_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    diag_input_data.update_input_desc_x(tensorDesc);
    diag_input_data.update_output_desc_y(tensorDesc);
    auto diag_op = op::Diag("diag_0");
    diag_op.set_input_x(diag_input_data);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(diag_op);
    std::vector<Operator> inputs{diag_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("DiagFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");

    bool findDiagD = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{3, 32, 3, 32};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DiagD") {
            findDiagD = true;
            auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
            std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
//        assert(findDiagD == true);
//        assert(shapeMatch == true);
    EXPECT_EQ(findDiagD, true);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(diag_fusion_test, diag_fusion_test_3) {
    ge::Graph graph("diag_fusion_test_3");
    auto diag_input_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    diag_input_data.update_input_desc_x(tensorDesc);
    diag_input_data.update_output_desc_y(tensorDesc);
    auto diag_op = op::Diag("diag_0");
    diag_op.set_input_x(diag_input_data);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(diag_op);
    std::vector<Operator> inputs{diag_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("DiagFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");

    bool findDiagD = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{3, 32, 3, 32};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DiagD") {
            findDiagD = true;
            auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
            std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
//        assert(findDiagD == true);
//        assert(shapeMatch == true);
    EXPECT_EQ(findDiagD, true);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(diag_fusion_test, diag_fusion_test_4) {
  ge::Graph graph("diag_fusion_test_4");
  auto diag_input_data = op::Data("diag_input_data");
  std::vector<int64_t> dims{1, 21, 34, 34};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  diag_input_data.update_input_desc_x(tensorDesc);
  diag_input_data.update_output_desc_y(tensorDesc);
  auto diag_op = op::Diag("diag_0");
  diag_op.set_input_x(diag_input_data);
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(diag_op);
  std::vector<Operator> inputs{diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("DiagFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");

  bool findDiagD = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 21, 34, 34, 1, 21, 34, 34};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "DiagD") {
      findDiagD = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
//        assert(findDiagD == true);
//        assert(shapeMatch == true);
  EXPECT_EQ(findDiagD, true);
  EXPECT_EQ(shapeMatch, true);
}
