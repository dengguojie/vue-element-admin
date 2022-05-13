//
// Created by c30002892 on 2020/9/5.
//

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class const2attr_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(const2attr_fusion_pass_test, const2attr_fusion_pass_test_1) {
  ge::Graph graph("const2attr_fusion_pass_test_1");
  auto diag_input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape);
  diag_input_data.update_input_desc_x(tensorDesc);
  diag_input_data.update_output_desc_y(tensorDesc);

  auto diag_input_data_size = op::Data("input_data_size");
  std::vector<int64_t> dims_size{3, 32};
  ge::Shape shape_size(dims_size);
  ge::TensorDesc tensorDescSize(shape_size);
  diag_input_data_size.update_input_desc_x(tensorDescSize);
  diag_input_data_size.update_output_desc_y(tensorDescSize);

  auto diag_op = op::ResizeBilinearV2("diag_0");
  diag_op.set_input_x(diag_input_data);
  diag_op.set_input_size(diag_input_data_size);
  diag_op.set_attr_align_corners(false);
  diag_op.set_attr_half_pixel_centers(false);

  auto end_op = op::Data("end_op_0");
  end_op.set_input_x(diag_op);

  std::vector<Operator> inputs{diag_input_data, diag_input_data_size};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
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
//  EXPECT_EQ(findDiagD, true);
//  EXPECT_EQ(shapeMatch, true);
}

TEST_F(const2attr_fusion_pass_test, const2attr_fusion_pass_test_2) {
  ge::Graph graph("const2attr_fusion_pass_test_2");
  auto diag_input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, -1};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape);
  diag_input_data.update_input_desc_x(tensorDesc);
  diag_input_data.update_output_desc_y(tensorDesc);

  auto diag_input_data_size = op::Data("input_data_size");
  std::vector<int64_t> dims_size{3, 32};
  ge::Shape shape_size(dims_size);
  ge::TensorDesc tensorDescSize(shape_size);
  diag_input_data_size.update_input_desc_x(tensorDescSize);
  diag_input_data_size.update_output_desc_y(tensorDescSize);

  auto diag_op = op::ResizeBilinearV2("diag_0");
  diag_op.set_input_x(diag_input_data);
  diag_op.set_input_size(diag_input_data_size);
  diag_op.set_attr_align_corners(false);
  diag_op.set_attr_half_pixel_centers(false);

  auto end_op = op::Data("end_op_0");
  end_op.set_input_x(diag_op);

  std::vector<Operator> inputs{diag_input_data, diag_input_data_size};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
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
}
