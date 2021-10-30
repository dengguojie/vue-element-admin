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
  matmul_input_data2.update_input_desc_x(tensorDesc_a);
  matmul_input_data2.update_output_desc_y(tensorDesc_a);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);


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
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test2) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test2");
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
  matmul_input_data2.update_input_desc_x(tensorDesc_a);
  matmul_input_data2.update_output_desc_y(tensorDesc_a);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);

  std::vector<int64_t> dims{1, 4, 4};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto data_bias = op::Constant();
  Tensor consttensor;
  float * dataValue = new float[1];
  * dataValue = 0.1;
  consttensor.SetTensorDesc(TensorDesc(ge::Shape({4}), FORMAT_NHWC));
  consttensor.SetData((uint8_t*)dataValue, 4);
  data_bias.set_attr_value(consttensor);

  auto add = op::Add("add")
        .set_input_x1(reshape_op)
        .set_input_x2(data_bias);

  std::vector<Operator> inputs{matmul_input_data1, matmul_input_data2, data_bias};
  std::vector<Operator> outputs{add};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test3) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test3");
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
  matmul_input_data2.update_input_desc_x(tensorDesc_a);
  matmul_input_data2.update_output_desc_y(tensorDesc_a);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);


  std::vector<int64_t> dims{1, 4, 4};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto data_bias = op::Constant();
  Tensor consttensor;
  float * dataValue = new float[1];
  * dataValue = 0.1;
  consttensor.SetTensorDesc(TensorDesc(ge::Shape({4}), FORMAT_NHWC));
  consttensor.SetData((uint8_t*)dataValue, 4);
  data_bias.set_attr_value(consttensor);

  auto add = op::Add("add")
        .set_input_x1(data_bias)
        .set_input_x2(reshape_op);

  std::vector<Operator> inputs{matmul_input_data1, matmul_input_data2, data_bias};
  std::vector<Operator> outputs{add};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test4) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test4");
  std::vector<int64_t> dims_a{4, 4};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data1 = op::Data("matmul_input_data1");
  matmul_input_data1.update_input_desc_x(tensorDesc_a);
  matmul_input_data1.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims_b{4, 6};
  ge::Shape shape_b(dims_b);
  ge::TensorDesc tensorDesc_b(shape_b, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data2 = op::Data("matmul_input_data2").set_attr_index(1);
  matmul_input_data2.update_input_desc_x(tensorDesc_a);
  matmul_input_data2.update_output_desc_y(tensorDesc_a);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);


  std::vector<int64_t> dims{2, 2, 6};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto bias_shape = vector<int64_t>({6});
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
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}

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
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool add_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Add") {
      add_match = true;
    }
  }
  EXPECT_EQ(add_match, false);
}

TEST_F(matmul_reshape_biasadd_fusion_test, matmul_reshape_biasadd_fusion_test5) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("matmul_reshape_biasadd_fusion_test5");
  std::vector<int64_t> dims_a{4, 4};
  ge::Shape shape_a(dims_a);
  ge::TensorDesc tensorDesc_a(shape_a, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data1 = op::Data("matmul_input_data1");
  matmul_input_data1.update_input_desc_x(tensorDesc_a);
  matmul_input_data1.update_output_desc_y(tensorDesc_a);

  std::vector<int64_t> dims_b{4, 6};
  ge::Shape shape_b(dims_b);
  ge::TensorDesc tensorDesc_b(shape_b, ge::FORMAT_ND, ge::DT_FLOAT);
  auto matmul_input_data2 = op::Data("matmul_input_data2").set_attr_index(1);
  matmul_input_data2.update_input_desc_x(tensorDesc_a);
  matmul_input_data2.update_output_desc_y(tensorDesc_a);

  auto matmul_op = ge::op::MatMul("matmul")
                                 .set_input_x1(matmul_input_data1)
                                 .set_input_x2(matmul_input_data2)
                                 .set_attr_transpose_x1(false)
                                 .set_attr_transpose_x2(false);


  std::vector<int64_t> dims{2, 2, 6};
  ge::Shape reshape_shape(dims);
  ge::TensorDesc tensorDesc_reshape(reshape_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  auto reshape_op = op::Reshape("reshape")
                               .set_input_x(matmul_op);
  reshape_op.update_output_desc_y(tensorDesc_reshape);

  auto bias_shape = vector<int64_t>({6});
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
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}