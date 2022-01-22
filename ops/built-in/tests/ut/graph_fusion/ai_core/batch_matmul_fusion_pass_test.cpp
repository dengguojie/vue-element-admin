#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "batch_matmul_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_matmul_fusion_test TearDown" << std::endl;
  }
};

TEST_F(batch_matmul_fusion_test, batch_matmul_fusion_test_1) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("batch_matmul_fusion_test_1");
  auto batch_matmul_input_data1 = op::Data("batch_matmul_input_data1");
  auto batch_matmul_input_data2 = op::Data("batch_matmul_input_data2");

  std::vector<int64_t> dims{3, 4};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

  std::vector<int64_t> dims1{4, 4};
  ge::Shape shape1(dims1);
  ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto batch_matmul_op = op::BatchMatMul("batch_matmul")
                             .set_input_x1(batch_matmul_input_data1)
                             .set_input_x2(batch_matmul_input_data2)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);

  batch_matmul_op.update_input_desc_x1(tensorDesc);
  batch_matmul_op.update_input_desc_x2(tensorDesc1);
  batch_matmul_op.update_output_desc_y(tensorDesc);

  std::vector<Operator> inputs{batch_matmul_input_data1, batch_matmul_input_data2};
  std::vector<Operator> outputs{batch_matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_left_1) {
  ge::Graph graph("bmm_transpose_left_1");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm0_value{0, 1, 3, 2};
  std::vector<int64_t> dims_transpose0{2, 3, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_right_bias_2) {
  ge::Graph graph("bmm_transpose_right_bias_2");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(true);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  auto dims_bias = vector<int64_t>({dims_y[dims_y.size() - 1]});
  ge::TensorDesc bias_desc(ge::Shape(dims_bias), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto bias_data = op::Data("bias");
  bias_data.update_input_desc_x(bias_desc);
  bias_data.update_output_desc_y(bias_desc);
  auto bias_op = op::BiasAdd("bias_add")
                              .set_input_x(matmul_op)
                              .set_input_bias(bias_data)
                              .set_attr_data_format("ND");

  std::vector<Operator> inputs{x1_data, x2_data, bias_data};
  std::vector<Operator> outputs{bias_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_both_3) {
  ge::Graph graph("bmm_transpose_both_3");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm0_value{1, 0};
  std::vector<int64_t> dims_transpose0{15, 32};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(true);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  auto relu_op = op::Relu("Relu").set_input_x(matmul_op);
  relu_op.update_input_desc_x(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{relu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1 and attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_left_4) {
  ge::Graph graph("bmmV2_transpose_left_4");

  std::vector<int64_t> dims_x1{3, 15, 32};
  std::vector<int64_t> dims_x2{2, 3, 20, 32};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm0_value{0, 2, 1};
  std::vector<int64_t> dims_transpose0{3, 32, 15};
  std::vector<int64_t> perm1_value{0, 1, 3, 2};
  std::vector<int64_t> dims_transpose1{2, 3, 32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  auto relu_op = op::Relu("Relu").set_input_x(transpose_1_op);
  relu_op.update_input_desc_x(desc_transpose1);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op, relu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      if (attr_transpose1 == false and x1_shape == dims_x1) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 1);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_right_5) {
  ge::Graph graph("bmmV2_transpose_right_5");

  std::vector<int64_t> dims_x1{3, 2, 15, 32};
  std::vector<int64_t> dims_x2{3, 20, 32};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm0_value{1, 0, 3, 2};
  std::vector<int64_t> dims_transpose0{2, 3, 32, 15};
  std::vector<int64_t> perm1_value{0, 2, 1};
  std::vector<int64_t> dims_transpose1{3, 32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose2 == true and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 1);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_both_6) {
  ge::Graph graph("bmmV2_transpose_both_6");

  std::vector<int64_t> dims_x1{2, 15, 32};
  std::vector<int64_t> dims_x2{2, 20, 32};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> perm0_value{0, 2, 1};
  std::vector<int64_t> dims_transpose0{2, 32, 15};
  std::vector<int64_t> perm1_value{0, 2, 1};
  std::vector<int64_t> dims_transpose1{2, 32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose1 == false and x1_shape == dims_x1 and attr_transpose2 == true and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmul_transpose_left_7) {
  ge::Graph graph("matmul_transpose_left_7");

  std::vector<int64_t> dims_x1{15, 32};
  std::vector<int64_t> dims_x2{20, 32};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm0_value{1, 0};
  std::vector<int64_t> dims_transpose0{32, 15};
  std::vector<int64_t> perm1_value{0, 1};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMul("MatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      if (attr_transpose1 == false and x1_shape == dims_x1) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 1);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmul_transpose_right_8) {
  ge::Graph graph("matmul_transpose_right_8");

  std::vector<int64_t> dims_x1{15, 32};
  std::vector<int64_t> dims_x2{20, 32};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto relu_op = op::Relu("Relu").set_input_x(x1_data);
  relu_op.update_input_desc_x(desc_x1);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMul("MatMul")
                       .set_input_x1(relu_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose2 == true and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmul_transpose_both_9) {
  ge::Graph graph("matmul_transpose_both_9");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm0_value{1, 0};
  std::vector<int64_t> dims_transpose0{15, 32};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMul("MatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(true);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1 and attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmulV2_transpose_left_10) {
  ge::Graph graph("matmulV2_transpose_left_10");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm0_value{1, 0};
  std::vector<int64_t> dims_transpose0{15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::MatMulV2("MatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmulV2_transpose_right_bias_11) {
  ge::Graph graph("matmulV2_transpose_right_bias_11");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMulV2("MatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(true);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  auto dims_bias = vector<int64_t>({dims_y[dims_y.size() - 1]});
  ge::TensorDesc bias_desc(ge::Shape(dims_bias), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto bias_data = op::Data("bias");
  bias_data.update_input_desc_x(bias_desc);
  bias_data.update_output_desc_y(bias_desc);
  auto bias_op = op::BiasAdd("bias_add")
                              .set_input_x(matmul_op)
                              .set_input_bias(bias_data)
                              .set_attr_data_format("ND");

  std::vector<Operator> inputs{x1_data, x2_data, bias_data};
  std::vector<Operator> outputs{bias_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmulV2_transpose_both_12) {
  ge::Graph graph("matmulV2_transpose_both_12");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{15, 20};
  std::vector<int64_t> perm0_value{1, 0};
  std::vector<int64_t> dims_transpose0{15, 32};
  std::vector<int64_t> perm1_value{1, 0};
  std::vector<int64_t> dims_transpose1{20, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose0.SetOriginShape(shape_transpose0);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMulV2("MatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(true);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1 and attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_dyn_left_1) {
  ge::Graph graph("bmm_transpose_dyn_left_1");

  std::vector<int64_t> dims_x1{2, -1, 32, 15};
  std::vector<int64_t> dims_x2{2, -1, 32, 20};
  std::vector<int64_t> dims_y{2, -1, 15, 20};
  std::vector<int64_t> dims_transpose0{2, -1, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // const perm
  auto shape_perm0 = ge::Shape({4});
  TensorDesc desc_perm0(shape_perm0, FORMAT_ND, DT_INT64);
  Tensor const_tensor0(desc_perm0);
  uint64_t *const_perm0_value = new uint64_t[4]{0, 1, 3, 2};
  const_tensor0.SetData((uint8_t *) const_perm0_value, 4 * sizeof(uint64_t));
  auto perm0_const = op::Const("perm0").set_attr_value(const_tensor0);
  delete[] const_perm0_value;

  auto transpose_0_op = op::Transpose("Transpose_0")
                            .set_input_x(x1_data)
                            .set_input_perm(perm0_const);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_input_desc_perm(desc_perm0);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm0_const};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "BatchMatMul") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose1;
      ge::AttrUtils::GetBool(matmul_desc, "adj_x1", attr_transpose1);
      vector<int64_t> x1_shape = matmul_desc->GetInputDesc(0).GetShape().GetDims();
      if (attr_transpose1 == true and x1_shape == dims_x1) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, matmulV2_transpose_dyn_right_2) {
  ge::Graph graph("matmulV2_transpose_dyn_right_2");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{32, -1};
  std::vector<int64_t> dims_y{15, -1};
  std::vector<int64_t> dims_transpose1{-1, 32};
  std::vector<std::pair<int64_t, int64_t>> range_x2 = {{10, 30}, {32, 32}};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose1.SetOriginShape(shape_transpose1);
  desc_transpose1.SetShapeRange(range_x2);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // const perm
  auto shape_perm1 = ge::Shape({2});
  TensorDesc desc_perm1(shape_perm1, FORMAT_ND, DT_INT32);
  Tensor const_tensor1(desc_perm1);
  uint32_t *const_perm1_value = new uint32_t[2]{1, 0};
  const_tensor1.SetData((uint8_t *) const_perm1_value, 2 * sizeof(uint32_t));
  auto perm1_const = op::Const("perm1").set_attr_value(const_tensor1);
  delete[] const_perm1_value;

  auto transpose_1_op = op::Transpose("Transpose_1")
                            .set_input_x(x2_data)
                            .set_input_perm(perm1_const);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_input_desc_perm(desc_perm1);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::MatMulV2("MatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(transpose_1_op)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(true);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm1_const};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  int find_transpose = 0;
  bool check_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose" or node->GetType() == "TransposeD") {
      find_transpose += 1;
    } else if (node->GetType() == "MatMulV2") {
      ge::OpDescPtr matmul_desc = node->GetOpDesc();
      bool attr_transpose2;
      ge::AttrUtils::GetBool(matmul_desc, "transpose_x2", attr_transpose2);
      vector<int64_t> x2_shape = matmul_desc->GetInputDesc(1).GetShape().GetDims();
      if (attr_transpose2 == false and x2_shape == dims_x2) {
        check_matmul = true;
      }
    }
  }
  EXPECT_EQ(find_transpose, 0);
  EXPECT_EQ(check_matmul, true);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_exception_2output_1) {
  ge::Graph graph("bmmV2_transpose_exception_2output_1");

  std::vector<int64_t> dims_x1{3, 15, 32};
  std::vector<int64_t> dims_x2{2, 3, 20, 32};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm1_value{0, 1, 3, 2};
  std::vector<int64_t> dims_transpose1{2, 3, 32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data)
                            .set_attr_perm(perm1_value);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  auto relu_op = op::Relu("Relu").set_input_x(transpose_1_op);
  relu_op.update_input_desc_x(desc_transpose1);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op, relu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_exception_transpose_2) {
  ge::Graph graph("bmmV2_transpose_exception_transpose_2");

  std::vector<int64_t> dims_x1{3, 2, 15, 32};
  std::vector<int64_t> dims_x2{3, 20, 32};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm0_value{1, 0, 3, 2};
  std::vector<int64_t> dims_transpose0{2, 3, 32, 15};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_exception_dtype_3) {
  ge::Graph graph("bmmV2_transpose_exception_dtype_3");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> perm0_value{0, 1, 3, 2};
  std::vector<int64_t> dims_transpose0{2, 3, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_INT8);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_INT8);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_INT32);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_INT8);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_0_op = op::TransposeD("TransposeD_0")
                            .set_input_x(x1_data)
                            .set_attr_perm(perm0_value);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_exception_perm_data_4) {
  ge::Graph graph("bmmV2_transpose_exception_perm_data_4");

  std::vector<int64_t> dims_x1{2, 3, -1, 15};
  std::vector<int64_t> dims_x2{2, 3, -1, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> dims_transpose0{2, 3, 15, -1};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // data perm
  ge::Shape shape_perm0({4});
  TensorDesc desc_perm0(shape_perm0, FORMAT_ND, DT_INT64);
  auto perm0_data = op::Data("perm0");

  auto transpose_0_op = op::Transpose("Transpose_0")
                            .set_input_x(x1_data)
                            .set_input_perm(perm0_data);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_input_desc_perm(desc_perm0);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm0_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_exception_perm_data_5) {
  ge::Graph graph("bmm_transpose_exception_perm_data_5");

  std::vector<int64_t> dims_x1{2, -1, 32, 15};
  std::vector<int64_t> dims_x2{2, -1, 32, 20};
  std::vector<int64_t> dims_y{2, -1, 15, 20};
  std::vector<int64_t> dims_transpose0{2, -1, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // const perm
  auto shape_perm0 = ge::Shape({4});
  TensorDesc desc_perm0(shape_perm0, FORMAT_ND, DT_INT8);
  Tensor const_tensor0(desc_perm0);
  uint8_t *const_perm0_value = new uint8_t[4]{0, 1, 3, 2};
  const_tensor0.SetData((uint8_t *) const_perm0_value, 4 * sizeof(uint8_t));
  auto perm0_const = op::Const("perm0").set_attr_value(const_tensor0);
  delete[] const_perm0_value;

  auto transpose_0_op = op::Transpose("Transpose_0")
                            .set_input_x(x1_data)
                            .set_input_perm(perm0_const);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_input_desc_perm(desc_perm0);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm0_const};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_exception_perm_data_6) {
  ge::Graph graph("bmm_transpose_exception_perm_data_6");

  std::vector<int64_t> dims_x1{2, -1, 32, 15};
  std::vector<int64_t> dims_x2{2, -1, 32, 20};
  std::vector<int64_t> dims_y{2, -1, 15, 20};
  std::vector<int64_t> dims_transpose0{2, -1, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // const perm
  auto shape_perm0 = ge::Shape({4});
  TensorDesc desc_perm0(shape_perm0, FORMAT_ND, DT_INT32);
  auto perm0_const = op::Const("perm0");

  auto transpose_0_op = op::Transpose("Transpose_0")
                            .set_input_x(x1_data)
                            .set_input_perm(perm0_const);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_input_desc_perm(desc_perm0);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm0_const};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmm_transpose_exception_perm_shape_7) {
  ge::Graph graph("bmm_transpose_exception_perm_shape_7");

  std::vector<int64_t> dims_x1{2, -1, 32, 15};
  std::vector<int64_t> dims_x2{2, -1, 32, 20};
  std::vector<int64_t> dims_y{2, -1, 15, 20};
  std::vector<int64_t> dims_transpose0{2, -1, 15, 32};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_transpose0(dims_transpose0);
  ge::TensorDesc desc_transpose0(shape_transpose0, FORMAT_ND, DT_FLOAT16);
  desc_transpose0.SetOriginShape(shape_transpose0);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");

  // const perm
  auto shape_perm0 = ge::Shape({1, 4});
  TensorDesc desc_perm0(shape_perm0, FORMAT_ND, DT_INT32);
  Tensor const_tensor0(desc_perm0);
  uint32_t *const_perm0_value = new uint32_t[4]{0, 1, 3, 2};
  const_tensor0.SetData((uint8_t *) const_perm0_value, 4 * sizeof(uint32_t));
  auto perm0_const = op::Const("perm0").set_attr_value(const_tensor0);
  delete[] const_perm0_value;

  auto transpose_0_op = op::Transpose("Transpose_0")
                            .set_input_x(x1_data)
                            .set_input_perm(perm0_const);
  transpose_0_op.update_input_desc_x(desc_x1);
  transpose_0_op.update_input_desc_perm(desc_perm0);
  transpose_0_op.update_output_desc_y(desc_transpose0);

  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(transpose_0_op)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(false)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_transpose0);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data, perm0_const};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(batch_matmul_fusion_test, bmmV2_transpose_exception_perm_list_8) {
  ge::Graph graph("bmmV2_transpose_exception_perm_list_8");

  std::vector<int64_t> dims_x1{3, 15, 32};
  std::vector<int64_t> dims_x2{2, 3, 20, 32};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> dims_transpose1{2, 3, 32, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Shape shape_transpose1(dims_transpose1);
  ge::TensorDesc desc_transpose1(shape_transpose1, FORMAT_ND, DT_FLOAT);
  desc_transpose1.SetOriginShape(shape_transpose1);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto transpose_1_op = op::TransposeD("TransposeD_1")
                            .set_input_x(x2_data);
  transpose_1_op.update_input_desc_x(desc_x2);
  transpose_1_op.update_output_desc_y(desc_transpose1);

  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(transpose_1_op)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_transpose1);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}