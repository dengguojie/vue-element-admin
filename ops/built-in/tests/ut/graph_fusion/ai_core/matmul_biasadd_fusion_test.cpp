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

namespace{
  static const char* BIASADD = "BiasAdd";
  static const char* ADD = "Add";
}
class matmul_biasadd_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "matmul_biasadd_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmul_biasadd_fusion_test TearDown" << std::endl;
  }
};

// Test Batch MatMul BiasAdd fusion Success(Functional)
TEST_F(matmul_biasadd_fusion_test, batchMatMul_BiasAdd_fusion_success) {
  ge::Graph graph("batchMatMul_BiasAdd_fusion_test");
  // Construt input Data 1
  std::vector<int64_t> dims1{2, 2, 16, 16};
  ge::Shape shapeA(dims1);
  ge::TensorDesc tensorDescA(shapeA, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData1 = op::Data("BatchMatMul_input_data1");
  inputData1.update_input_desc_x(tensorDescA);
  inputData1.update_output_desc_y(tensorDescA);
  // Construt input Data 2
  std::vector<int64_t> dimsB{2, 2, 16, 1024};
  ge::Shape shapeB(dimsB);
  ge::TensorDesc tensorDescB(shapeB, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData2 = op::Data("BatchMatMul_input_data2");
  inputData2.update_input_desc_x(tensorDescB);
  inputData2.update_output_desc_y(tensorDescB);
  // Construct BatchMatMul and BiasAdd fusion
  auto batchMatMulOp = ge::op::BatchMatMul("BatchMatMul")
                                          .set_input_x1(inputData1)
                                          .set_input_x2(inputData2)
                                          .set_attr_adj_x1(false)
                                          .set_attr_adj_x2(false);
  auto bias_shape = vector<int64_t>({1024});
  ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto data_bias = op::Data("data_bias");
  data_bias.update_input_desc_x(bias_desc);
  data_bias.update_output_desc_y(bias_desc);

  auto biasAdd = op::BiasAdd("bias_add")
                              .set_input_x(batchMatMulOp)
                              .set_input_bias(data_bias)
                              .set_attr_data_format("NHWC");
  // Set Graph and Expected Res
  std::vector<Operator> inputs{inputData1, inputData2, data_bias};
  std::vector<Operator> outputs{biasAdd};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool fusionSuccess = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == BIASADD) {
      fusionSuccess = false;
    }
  }
  EXPECT_EQ(fusionSuccess, true);
}

// Test Batch MatMul Add Fusion Success(Functional)
TEST_F(matmul_biasadd_fusion_test, batchMatMul_Add_fusion_success) {
  ge::Graph graph("batchMatMul_Add_fusion_test");
  // Construt input Data 1
  std::vector<int64_t> dims1{2, 2, 16, 16};
  ge::Shape shapeA(dims1);
  ge::TensorDesc tensorDescA(shapeA, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData1 = op::Data("BatchMatMul_input_data1");
  inputData1.update_input_desc_x(tensorDescA);
  inputData1.update_output_desc_y(tensorDescA);
  // Construt input Data 2
  std::vector<int64_t> dimsB{2, 2, 16, 1024};
  ge::Shape shapeB(dimsB);
  ge::TensorDesc tensorDescB(shapeB, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData2 = op::Data("BatchMatMul_input_data2");
  inputData2.update_input_desc_x(tensorDescB);
  inputData2.update_output_desc_y(tensorDescB);
  // Construct BatchMatMul and BiasAdd fusion
  auto batchMatMulOp = ge::op::BatchMatMul("BatchMatMul")
                                          .set_input_x1(inputData1)
                                          .set_input_x2(inputData2)
                                          .set_attr_adj_x1(false)
                                          .set_attr_adj_x2(false);
  auto bias_shape = vector<int64_t>({1024});
  ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto data_bias = op::Data("add_node");
  data_bias.update_input_desc_x(bias_desc);
  data_bias.update_output_desc_y(bias_desc);

  auto addOp = op::Add("add_op")
                      .set_input_x1(batchMatMulOp)
                      .set_input_x2(data_bias);
  // Set Graph and Expected Res
  std::vector<Operator> inputs{inputData1, inputData2, data_bias};
  std::vector<Operator> outputs{addOp};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool fusionSuccess = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == ADD) {
      fusionSuccess = false;
    }
  }
  EXPECT_EQ(fusionSuccess, true);
}

// Test Batch MatMul Add Fusion Success2(Functional)
TEST_F(matmul_biasadd_fusion_test, batchMatMul_Add_fusion_success2) {
  ge::Graph graph("batchMatMul_Add_fusion_test2");
  // Construt input Data 1
  std::vector<int64_t> dims1{2, 2, 16, 16};
  ge::Shape shapeA(dims1);
  ge::TensorDesc tensorDescA(shapeA, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData1 = op::Data("BatchMatMul_input_data1");
  inputData1.update_input_desc_x(tensorDescA);
  inputData1.update_output_desc_y(tensorDescA);
  // Construt input Data 2
  std::vector<int64_t> dimsB{2, 2, 16, 1024};
  ge::Shape shapeB(dimsB);
  ge::TensorDesc tensorDescB(shapeB, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData2 = op::Data("BatchMatMul_input_data2");
  inputData2.update_input_desc_x(tensorDescB);
  inputData2.update_output_desc_y(tensorDescB);
  // Construct BatchMatMul and BiasAdd fusion
  auto batchMatMulOp = ge::op::BatchMatMul("BatchMatMul")
                                          .set_input_x1(inputData1)
                                          .set_input_x2(inputData2)
                                          .set_attr_adj_x1(false)
                                          .set_attr_adj_x2(false);
  auto bias_shape = vector<int64_t>({1024});
  ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto data_bias = op::Data("add_node");
  data_bias.update_input_desc_x(bias_desc);
  data_bias.update_output_desc_y(bias_desc);

  auto addOp = op::Add("add_op")
                      .set_input_x1(data_bias)
                      .set_input_x2(batchMatMulOp);
  // Set Graph and Expected Res
  std::vector<Operator> inputs{inputData1, inputData2, data_bias};
  std::vector<Operator> outputs{addOp};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool fusionSuccess = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == ADD) {
      fusionSuccess = false;
    }
  }
  EXPECT_EQ(fusionSuccess, true);
}

// Test Batch MatMul Add Fusion Not Fused Senario(Function)
TEST_F(matmul_biasadd_fusion_test, batchMatMul_Add_no_fusion) {
  ge::Graph graph("batchMatMul_Add_fusion_test2");
  // Construt input Data 1
  std::vector<int64_t> dims1{2, 2, 16, 16};
  ge::Shape shapeA(dims1);
  ge::TensorDesc tensorDescA(shapeA, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData1 = op::Data("BatchMatMul_input_data1");
  inputData1.update_input_desc_x(tensorDescA);
  inputData1.update_output_desc_y(tensorDescA);
  // Construt input Data 2
  std::vector<int64_t> dimsB{2, 2, 16, 1024};
  ge::Shape shapeB(dimsB);
  ge::TensorDesc tensorDescB(shapeB, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto inputData2 = op::Data("BatchMatMul_input_data2");
  inputData2.update_input_desc_x(tensorDescB);
  inputData2.update_output_desc_y(tensorDescB);
  // Construct BatchMatMul and BiasAdd fusion
  auto batchMatMulOp = ge::op::BatchMatMul("BatchMatMul")
                                          .set_input_x1(inputData1)
                                          .set_input_x2(inputData2)
                                          .set_attr_adj_x1(false)
                                          .set_attr_adj_x2(false);
  auto add_shape = vector<int64_t>({2, 2, 16, 1024});
  ge::TensorDesc add_desc(ge::Shape(add_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto dataToAdd = op::Data("add_node");
  dataToAdd.update_input_desc_x(add_desc);
  dataToAdd.update_output_desc_y(add_desc);

  auto addOp = op::Add("add_op")
                      .set_input_x1(batchMatMulOp)
                      .set_input_x2(dataToAdd);
  // Set Graph and Expected Res
  std::vector<Operator> inputs{inputData1, inputData2, dataToAdd};
  std::vector<Operator> outputs{addOp};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool fusionSuccess = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == ADD) {
      fusionSuccess = false;
    }
  }
  EXPECT_EQ(fusionSuccess, false);
}