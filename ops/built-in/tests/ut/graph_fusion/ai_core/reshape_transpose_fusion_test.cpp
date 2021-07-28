#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class reshape_transpose_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "reshape_transpose_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "reshape_transpose_fusion_test TearDown" << std::endl;
    }
};

TEST_F(reshape_transpose_fusion_test, reshape_transpose_fusion_test_base1) {
  ge::Graph graph("reshape_transpose_fusion_test_base1");
  std::vector<int64_t> dims_a{8192, 1024};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_data = op::Data("input_data");
  input_data.update_input_desc_x(tensorDesc_a);
  input_data.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims{16, 512, 16, 64};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensor_desc_reshape(reshape_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_reshape.SetOriginShape(reshape_shape);
  tensor_desc_reshape.SetOriginFormat(ge::FORMAT_ND);
  auto reshape_op = op::Reshape("reshape").set_input_x(input_data);
  reshape_op.update_output_desc_y(tensor_desc_reshape);

  auto transpose_op = op::TransposeD("TransposeD")
                    .set_input_x(reshape_op)
                    .set_attr_perm({0,2,1,3});
  std::vector<int64_t> transpose_dims{16, 16, 512, 64};
  ge::Shape transpose_shape(transpose_dims);
  ge::TensorDesc tensor_desc_transpose(transpose_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_transpose.SetOriginShape(reshape_shape);
  tensor_desc_transpose.SetOriginFormat(ge::FORMAT_ND);
  transpose_op.update_output_desc_y(tensor_desc_transpose);
  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transpose_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ReshapeTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool transpose_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      transpose_match = true;
    }
  }
  EXPECT_EQ(transpose_match, false);
}

TEST_F(reshape_transpose_fusion_test, reshape_transpose_fusion_test_base2_2transpose) {
  ge::Graph graph("reshape_transpose_fusion_test_base2_2transpose");
  std::vector<int64_t> dims_a{8192, 1024};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_data = op::Data("input_data");
  input_data.update_input_desc_x(tensorDesc_a);
  input_data.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims{16, 512, 16, 64};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensor_desc_reshape(reshape_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_reshape.SetOriginShape(reshape_shape);
  tensor_desc_reshape.SetOriginFormat(ge::FORMAT_ND);
  auto reshape_op = op::Reshape("reshape").set_input_x(input_data);
  reshape_op.update_output_desc_y(tensor_desc_reshape);

  auto transpose1_op = op::TransposeD("TransposeD1")
                    .set_input_x(reshape_op)
                    .set_attr_perm({0,2,1,3});
  std::vector<int64_t> transpose_dims{16, 16, 512, 64};
  ge::Shape transpose_shape(transpose_dims);
  ge::TensorDesc tensor_desc_transpose(transpose_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_transpose.SetOriginShape(reshape_shape);
  tensor_desc_transpose.SetOriginFormat(ge::FORMAT_ND);
  transpose1_op.update_output_desc_y(tensor_desc_transpose);

  auto transpose2_op = op::TransposeD("TransposeD2")
                    .set_input_x(reshape_op)
                    .set_attr_perm({0,2,1,3});
  transpose2_op.update_output_desc_y(tensor_desc_transpose);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transpose1_op, transpose2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ReshapeTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool transpose_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      transpose_match = true;
    }
  }
  EXPECT_EQ(transpose_match, false);
}
