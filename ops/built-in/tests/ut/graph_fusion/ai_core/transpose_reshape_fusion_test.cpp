#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class transpose_reshape_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "transpose_reshape_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transpose_reshape_fusion_test TearDown" << std::endl;
  }
};

TEST_F(transpose_reshape_fusion_test, transpose_reshape_fusion_test_1) {
  ge::Graph graph("transpose_reshape_fusion_test_1");
  std::vector<int64_t> dims_a{1000, 5, 64, 64};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_data = op::Data("input_data");
  input_data.update_input_desc_x(tensorDesc_a);
  input_data.update_output_desc_y(tensorDesc_a);

  auto transpose_op = op::TransposeD("TransposeD").set_input_x(input_data).set_attr_perm({0, 2, 1, 3});
  std::vector<int64_t> transpose_dims{1000, 64, 5, 64};
  ge::Shape transpose_shape(transpose_dims);
  ge::TensorDesc tensor_desc_transpose(transpose_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_transpose.SetOriginShape(transpose_shape);
  tensor_desc_transpose.SetOriginFormat(ge::FORMAT_ND);
  transpose_op.update_output_desc_y(tensor_desc_transpose);

  std::vector<int64_t> dims{1000, 64, 320};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensor_desc_reshape(reshape_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_reshape.SetOriginShape(reshape_shape);
  tensor_desc_reshape.SetOriginFormat(ge::FORMAT_ND);
  auto reshape_op = op::Reshape("reshape").set_input_x(transpose_op);
  reshape_op.update_output_desc_y(tensor_desc_reshape);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{reshape_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransposeReshapeFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool transpose_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    std::cout << "The node is: " << node->GetType() << std::endl;
    if (node->GetType() == "ConfusionTransposeD") {
      transpose_match = true;
    }
  }
  EXPECT_EQ(transpose_match, true);
}

TEST_F(transpose_reshape_fusion_test, transpose_reshape_fusion_test_2) {
  ge::Graph graph("transpose_reshape_fusion_test_2");
  std::vector<int64_t> dims_a{1000, 5, 64, 64};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_data = op::Data("input_data");
  input_data.update_input_desc_x(tensorDesc_a);
  input_data.update_output_desc_y(tensorDesc_a);

  auto transpose_op = op::TransposeD("TransposeD").set_input_x(input_data).set_attr_perm({0, 2, 1, 3});
  std::vector<int64_t> transpose_dims{1000, 64, 5, 64};
  ge::Shape transpose_shape(transpose_dims);
  ge::TensorDesc tensor_desc_transpose(transpose_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_transpose.SetOriginShape(transpose_shape);
  tensor_desc_transpose.SetOriginFormat(ge::FORMAT_ND);
  transpose_op.update_output_desc_y(tensor_desc_transpose);

  std::vector<int64_t> dims{1000, 64, 320};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensor_desc_reshape(reshape_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_reshape.SetOriginShape(reshape_shape);
  tensor_desc_reshape.SetOriginFormat(ge::FORMAT_ND);

  ge::Tensor shape_tensor;
  std::vector<int64_t> shape_vec{3};
  ge::Shape shape_shape(shape_vec);
  ge::TensorDesc shape_desc(shape_shape, FORMAT_ND, DT_INT32);
  int32_t shape_size = shape_desc.GetShape().GetShapeSize();
  shape_desc.SetSize(shape_size * sizeof(int32_t));
  shape_tensor.SetTensorDesc(shape_desc);
  int32_t* shape_data = nullptr;
  shape_data = new int32_t[shape_size];
  *(shape_data + 0) = 1000;
  *(shape_data + 1) = 64;
  *(shape_data + 2) = 320;
  shape_tensor.SetData((uint8_t*)shape_data, shape_size * sizeof(int32_t));
  delete[] shape_data;

  // auto shape_op = op::Shape("shape").set_input_x(transpose_op);
  // auto reshapeConst = op::Constant().set_attr_value(shape_op);
  auto reshapeConst = op::Constant().set_attr_value(shape_tensor);
  auto reshape_op = op::Reshape("reshape").set_input_x(transpose_op).set_input_shape(reshapeConst);
  reshape_op.update_output_desc_y(tensor_desc_reshape);

  // GraphUtils::AddEdge(transpose_op->GetOutControlAnchor(), reshape_op->GetInControlAnchor());

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{reshape_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransposeReshapeFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool transpose_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    std::cout << "The node is: " << node->GetType() << std::endl;
    if (node->GetType() == "ConfusionTransposeD") {
      transpose_match = true;
    }
  }
  EXPECT_EQ(transpose_match, true);
}
