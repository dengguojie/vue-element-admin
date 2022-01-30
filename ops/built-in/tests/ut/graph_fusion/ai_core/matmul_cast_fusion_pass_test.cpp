#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class matmul_cast_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "matmul_cast_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmul_cast_fusion_test TearDown" << std::endl;
  }
};

TEST_F(matmul_cast_fusion_test, bmm_cast_1) {
  ge::Graph graph("bmm_cast_1");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool find_cast = false;
  bool check_matmul_output_dtype = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      find_cast = true;
    } else if (node->GetType() == "BatchMatMul") {
      DataType matmul_output_dtype = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
      if (matmul_output_dtype == ge::DT_FLOAT) {
        check_matmul_output_dtype = true;
      }
    }
  }
  EXPECT_EQ(find_cast, false);
  EXPECT_EQ(check_matmul_output_dtype, true);
}

TEST_F(matmul_cast_fusion_test, bmmV2_cast_2) {
  ge::Graph graph("bmmV2_cast_2");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool find_cast = false;
  bool check_matmul_output_dtype = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      find_cast = true;
    } else if (node->GetType() == "BatchMatMulV2") {
      DataType matmul_output_dtype = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
      if (matmul_output_dtype == ge::DT_FLOAT) {
        check_matmul_output_dtype = true;
      }
    }
  }
  EXPECT_EQ(find_cast, false);
  EXPECT_EQ(check_matmul_output_dtype, true);
}

TEST_F(matmul_cast_fusion_test, matmul_cast_3) {
  ge::Graph graph("matmul_cast_3");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::MatMul("MatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  auto relu_op = op::Relu("Relu").set_input_x(cast_op);
  relu_op.update_input_desc_x(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{relu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool find_cast = false;
  bool check_matmul_output_dtype = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      find_cast = true;
    } else if (node->GetType() == "MatMul") {
      DataType matmul_output_dtype = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
      if (matmul_output_dtype == ge::DT_FLOAT) {
        check_matmul_output_dtype = true;
      }
    }
  }
  EXPECT_EQ(find_cast, false);
  EXPECT_EQ(check_matmul_output_dtype, true);
}

TEST_F(matmul_cast_fusion_test, matmulV2_cast_4) {
  ge::Graph graph("matmulV2_cast_4");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::MatMulV2("MatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool find_cast = false;
  bool check_matmul_output_dtype = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      find_cast = true;
    } else if (node->GetType() == "MatMulV2") {
      DataType matmul_output_dtype = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
      if (matmul_output_dtype == ge::DT_FLOAT) {
        check_matmul_output_dtype = true;
      }
    }
  }
  EXPECT_EQ(find_cast, false);
  EXPECT_EQ(check_matmul_output_dtype, true);
}

TEST_F(matmul_cast_fusion_test, bmm_cast_exception_2output_1) {
  ge::Graph graph("bmm_cast_exception_2output_1");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  auto relu_op = op::Relu("Relu").set_input_x(matmul_op);
  relu_op.update_input_desc_x(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op, relu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(matmul_cast_fusion_test, bmm_cast_exception_bmmdtype_2) {
  ge::Graph graph("bmm_cast_exception_bmmdtype_2");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(matmul_cast_fusion_test, bmm_cast_exception_castdtype_3) {
  ge::Graph graph("bmm_cast_exception_castdtype_3");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_INT32);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(matmul_cast_fusion_test, bmmV2_cast_exception_dyn_4) {
  ge::Graph graph("bmmV2_cast_exception_dyn_4");

  std::vector<int64_t> dims_x1{-2};
  std::vector<int64_t> dims_x2{2, 32, 200};
  std::vector<int64_t> dims_y{-2};
  std::vector<std::pair<int64_t, int64_t>> range_x2 = {{2, 2}, {32, 32}, {200, 200}};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetShapeRange(range_x2);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(matmul_cast_fusion_test, bmmV2_cast_exception_dyn_rank_5) {
  ge::Graph graph("bmmV2_cast_exception_dyn_rank_5");

  std::vector<int64_t> dims_x1{32, 100};
  std::vector<int64_t> dims_x2{2, 32, -1};
  std::vector<int64_t> dims_y{100, -1};
  std::vector<std::pair<int64_t, int64_t>> range_x2 = {{2, 2}, {32, 32}, {100, 200}};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetShapeRange(range_x2);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(matmul_cast_fusion_test, mm_cast_exception_dyn_bias_6) {
  ge::Graph graph("mm_cast_exception_dyn_bias_6");

  std::vector<int64_t> dims_x1{32, 100};
  std::vector<int64_t> dims_x2{32, 200};
  std::vector<int64_t> dims_y{100, 200};
  std::vector<int64_t> dims_bias{-1};
  std::vector<std::pair<int64_t, int64_t>> range_bias = {{100, 200}};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_bias(dims_bias);
  ge::TensorDesc desc_bias(shape_bias, FORMAT_ND, ge::DT_FLOAT16);
  desc_bias.SetShapeRange(range_bias);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc desc_cast(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto bias_data = op::Data("bias");
  auto matmul_op = op::MatMul("MatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_input_bias(bias_data)
                       .set_attr_transpose_x1(true)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_input_desc_bias(desc_bias);
  matmul_op.update_output_desc_y(desc_y);

  auto cast_op = op::Cast("Cast").set_input_x(matmul_op).set_attr_dst_type(ge::DT_FLOAT);
  cast_op.update_input_desc_x(desc_y);
  cast_op.update_output_desc_y(desc_cast);

  std::vector<Operator> inputs{x1_data, x2_data, bias_data};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("MatmulCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}
