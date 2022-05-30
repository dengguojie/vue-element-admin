#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class matmul_reshape_biasadd_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "matmul_reshape_biasadd_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "matmul_reshape_biasadd_fusion_test TearDown" << std::endl;
    }
};

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_add_fusion_test1) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_add_fusion_test1");
  std::vector<int64_t> dims_a{8192, 1024};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data1 = op::Data("matmul_input_data1");
  matmul_input_data1.update_input_desc_x(tensorDesc_a);
  matmul_input_data1.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims_b{1024, 1024};
  ge::Shape shape_b(dims_b);
  ge::TensorDesc tensorDesc_b(shape_b, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data2 = op::Data("matmul_input_data2").set_attr_index(1);
  matmul_input_data2.update_input_desc_x(tensorDesc_b);
  matmul_input_data2.update_output_desc_y(tensorDesc_b);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);

  std::vector<int64_t> shape_matmul_output{8192, 1024};
  ge::TensorDesc desc_matmul_output(ge::Shape(shape_matmul_output), ge::FORMAT_ND, ge::DT_FLOAT);
  matmul_op.update_output_desc_y(desc_matmul_output);

  std::vector<int64_t> dims{16, 512, 16, 64};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  int element_size = 16 * 64;
  auto bias_shape = vector<int64_t>({16, 64});
  ge::Tensor constTensor;
  ge::TensorDesc bias_desc(ge::Shape(bias_shape),
      ge::FORMAT_NHWC, ge::DT_FLOAT16);
  bias_desc.SetSize(element_size * sizeof(int32_t));
  constTensor.SetTensorDesc(bias_desc);

  int *bias_tensor_value = new int[element_size];
  for (int i = 0; i < element_size; i++) {
      *(bias_tensor_value + i) = 0;
  }
  constTensor.SetData((uint8_t *) bias_tensor_value,
    element_size * sizeof(int32_t));
  auto bias_op = ge::op::Constant("bias").set_attr_value(constTensor);
  bias_op.update_output_desc_y(bias_desc);

  auto add_op = op::Add("add")
        .set_input_x1(reshape_op)
        .set_input_x2(bias_op);

  std::vector<Operator> inputs{matmul_input_data1, matmul_input_data2};
  std::vector<Operator> outputs{add_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  Status res = fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass",
                                       fe::BUILT_IN_GRAPH_PASS,
                                       *compute_graph_ptr);
  EXPECT_EQ(res, SUCCESS);

  bool add_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Add") {
      add_match = true;
    }
  }
  EXPECT_EQ(add_match, false);
}

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test1) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test1");
  std::vector<int64_t> dims_a{4, 4};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data1 = op::Data("matmul_input_data1");
  matmul_input_data1.update_input_desc_x(tensorDesc_a);
  matmul_input_data1.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims_b{4, 4};
  ge::Shape shape_b(dims_b);
  ge::TensorDesc tensorDesc_b(shape_b, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data2 = op::Data("matmul_input_data2").set_attr_index(1);
  matmul_input_data2.update_input_desc_x(tensorDesc_b);
  matmul_input_data2.update_output_desc_y(tensorDesc_b);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);

  std::vector<int64_t> shape_matmul_output{4, 4};
  ge::TensorDesc desc_matmul_output(ge::Shape(shape_matmul_output), ge::FORMAT_ND, ge::DT_FLOAT);
  matmul_op.update_output_desc_y(desc_matmul_output);

  std::vector<int64_t> dims{1, 4, 4};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto bias_shape = vector<int64_t>({4});
  ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto data_bias = op::Data("data_bias");
  data_bias.update_input_desc_x(bias_desc);
  data_bias.update_output_desc_y(bias_desc);

  auto bias_add = op::BiasAdd("bias_add")
        .set_input_x(reshape_op)
        .set_input_bias(data_bias)
        .set_attr_data_format("NHWC");


  std::vector<Operator> inputs{matmul_input_data1, matmul_input_data2, data_bias};
  std::vector<Operator> outputs{bias_add};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  Status res = fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass",
                                       fe::BUILT_IN_GRAPH_PASS,
                                       *compute_graph_ptr);
  EXPECT_EQ(res, SUCCESS);
  
  bool biasadd_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BiasAdd") {
      biasadd_match = true;
    }
  }
  EXPECT_EQ(biasadd_match, false);
}


TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test2) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test_with_control_anchor");
  std::vector<int64_t> dims_a{4, 4};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT16);
  auto matmul_input_data1 = op::Data("matmul_input_data1");
  matmul_input_data1.update_input_desc_x(tensorDesc_a);
  matmul_input_data1.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims_b{4, 4};
  ge::Shape shape_b(dims_b);
  ge::TensorDesc tensorDesc_b(shape_b, ge::FORMAT_ND, ge::DT_FLOAT16);
  auto matmul_input_data2 = op::Data("matmul_input_data2").set_attr_index(1);
  matmul_input_data2.update_input_desc_x(tensorDesc_b);
  matmul_input_data2.update_output_desc_y(tensorDesc_b);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);

  std::vector<int64_t> shape_matmul_output{4, 4};
  ge::TensorDesc desc_matmul_output(ge::Shape(shape_matmul_output), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_op.update_output_desc_y(desc_matmul_output);

  std::vector<int64_t> dims_reshape{1, 4, 4};
  ge::Shape reshape_shape(dims_reshape);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto bias_shape = vector<int64_t>({4});
  ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto data_bias = op::Data("data_bias");
  data_bias.update_input_desc_x(bias_desc);
  data_bias.update_output_desc_y(bias_desc);

  auto bias_add = op::BiasAdd("bias_add")
        .set_input_x(reshape_op)
        .set_input_bias(data_bias)
        .set_attr_data_format("NHWC");

  std::vector<int64_t> shape_biasadd_output{1, 4, 4};
  ge::TensorDesc desc_biasadd_output(ge::Shape(shape_biasadd_output), ge::FORMAT_ND, ge::DT_FLOAT16);
  bias_add.update_output_desc_y(desc_biasadd_output);

  std::vector<int64_t> dims_cast{1, 4, 4};
  ge::Shape cast_shape(dims_cast);
  ge::TensorDesc desc_cast(cast_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  auto cast_op = op::Cast("Cast").set_input_x(bias_add).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(bias_desc);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{matmul_input_data1, matmul_input_data2, data_bias};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  NodePtr node1 = nullptr;
  NodePtr node2 = nullptr;
  NodePtr node3 = nullptr;
  NodePtr node4 = nullptr;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      node1 = node;
    }
    if (node->GetType() == "Reshape") {
      node2 = node;
    }
    if (node->GetType() == "BiasAdd") {
      node3 = node;
    }
    if (node->GetType() == "Cast") {
      node4 = node;
    }
  }
  GraphUtils::AddEdge(node1->GetOutControlAnchor(), node2->GetInControlAnchor());
  GraphUtils::AddEdge(node2->GetOutControlAnchor(), node3->GetInControlAnchor());
  GraphUtils::AddEdge(node3->GetOutControlAnchor(), node4->GetInControlAnchor());

  Status res = fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass",
                                       fe::BUILT_IN_GRAPH_PASS,
                                       *compute_graph_ptr);
  EXPECT_EQ(res, SUCCESS);
  
  bool biasadd_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BiasAdd") {
      biasadd_match = true;
    }
  }
  EXPECT_EQ(biasadd_match, false);
}