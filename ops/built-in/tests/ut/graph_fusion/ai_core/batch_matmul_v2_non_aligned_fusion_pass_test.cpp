#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "framework/common/types.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_v2_non_aligned_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "batch_matmul_v2_non_aligned_fusion_pass_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "batch_matmul_v2_non_aligned_fusion_pass_test TearDown" << std::endl; }
};

bool ExecuteCorrectly(ge::ComputeGraphPtr compute_graph_ptr) {
  bool res = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadD") {
      res = true;
    }
  }
  return res;
}

TEST_F(batch_matmul_v2_non_aligned_fusion_pass_test, batchmatmul_pattern_1) {
  std::cout << "enter batch_matmul_v2_non_aligned_fusion_pass_test.batchmatmul_pattern_1" << std::endl;
  ge::Graph graph("batchmatmul_pattern_1");
  ge::Shape x1_shape({1000, 48, 324});
  ge::Shape x2_shape({324, 324});
  ge::Shape add_const_shape({324});
  ge::Shape reshape_1_shape({1000, 48, 12, 27});
  ge::Shape transpose_1_shape({1000, 12, 48, 27});
  ge::Shape bmm2_input_1_shape({1000, 12, 48, 48});
  ge::Shape reshape_1_shape_shape({4});
  ge::Shape x1_shape_shape({3});

  ge::TensorDesc x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  x1_desc.SetOriginShape(x1_shape);
  auto data_x1 = op::Data("data_x1");
  data_x1.update_input_desc_x(x1_desc);
  data_x1.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor x2_tensor(x2_desc);
  auto const_x2 = op::Const().set_attr_value(x2_tensor);

  auto batchmatmul_1 = op::BatchMatMulV2("batchmatmul_1")
                           .set_input_x1(data_x1)
                           .set_input_x2(const_x2)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm1_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm1_input_x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm1_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  bmm1_input_x1_desc.SetOriginShape(x1_shape);
  bmm1_input_x2_desc.SetOriginShape(x2_shape);
  bmm1_output_y_desc.SetOriginShape(x1_shape);
  batchmatmul_1.update_input_desc_x1(bmm1_input_x1_desc);
  batchmatmul_1.update_input_desc_x2(bmm1_input_x2_desc);
  batchmatmul_1.update_output_desc_y(bmm1_output_y_desc);

  ge::TensorDesc add_const_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor add_const_tensor(add_const_desc);
  auto add_const = op::Const().set_attr_value(add_const_tensor);

  auto add_1 = op::Add("add_1").set_input_x1(add_const).set_input_x2(batchmatmul_1);

  ge::TensorDesc add1_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add1_input_x2_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add1_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  add1_input_x1_desc.SetOriginShape(x1_shape);
  add1_input_x2_desc.SetOriginShape(add_const_shape);
  add1_output_y_desc.SetOriginShape(x1_shape);
  add_1.update_input_desc_x1(add1_input_x1_desc);
  add_1.update_input_desc_x2(add1_input_x2_desc);
  add_1.update_output_desc_y(add1_output_y_desc);

  // reshape const
  TensorDesc desc_input_size_1(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  Tensor reshape_const_tensor(desc_input_size_1);
  uint32_t *reshape_const_tensor_value = new uint32_t[4];
  for (size_t dim = 0; dim < 4; dim++) {
    *(reshape_const_tensor_value + dim) = reshape_1_shape.GetDim(dim);
  }
  reshape_const_tensor.SetData((uint8_t *)reshape_const_tensor_value, 4 * sizeof(uint32_t));

  auto reshape_const = op::Const("reshape_1_const").set_attr_value(reshape_const_tensor);

  auto reshape_1 = op::Reshape("reshape_1").set_input_x(add_1).set_input_shape(reshape_const);
  ge::TensorDesc reshape_input_x_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc reshape_shape_desc(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  ge::TensorDesc reshape_output_y_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  reshape_input_x_desc.SetOriginShape(x1_shape);
  reshape_output_y_desc.SetOriginShape(reshape_1_shape);
  reshape_1.update_input_desc_x(reshape_input_x_desc);
  reshape_1.update_input_desc_shape(reshape_shape_desc);
  reshape_1.update_output_desc_y(reshape_output_y_desc);

  auto transpose_1 = op::TransposeD("transpose_1").set_input_x(reshape_1).set_attr_perm({0, 2, 1, 3});
  ge::TensorDesc transpose_input_x_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc transpose_output_y_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  transpose_input_x_desc.SetOriginShape(reshape_1_shape);
  transpose_output_y_desc.SetOriginShape(transpose_1_shape);
  transpose_1.update_input_desc_x(transpose_input_x_desc);
  transpose_1.update_output_desc_y(transpose_output_y_desc);

  ge::TensorDesc bmm2_x2_desc(bmm2_input_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor bmm2_x2_tensor(bmm2_x2_desc);
  auto bmm2_const_x2 = op::Const().set_attr_value(bmm2_x2_tensor);

  auto batchmatmul_2 = op::BatchMatMulV2("batchmatmul_2")
                           .set_input_x1(bmm2_const_x2)
                           .set_input_x2(transpose_1)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm2_input_x1_desc(bmm2_input_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm2_input_x2_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm2_output_y_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  bmm2_input_x1_desc.SetOriginShape(bmm2_input_1_shape);
  bmm2_input_x2_desc.SetOriginShape(transpose_1_shape);
  bmm2_output_y_desc.SetOriginShape(transpose_1_shape);
  batchmatmul_2.update_input_desc_x1(bmm2_input_x1_desc);
  batchmatmul_2.update_input_desc_x2(bmm2_input_x2_desc);
  batchmatmul_2.update_output_desc_y(bmm2_output_y_desc);

  auto transpose_2 = op::TransposeD("transpose_2").set_input_x(batchmatmul_2).set_attr_perm({0, 2, 1, 3});
  ge::TensorDesc transpose2_input_x_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc transpose2_output_y_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  transpose2_input_x_desc.SetOriginShape(transpose_1_shape);
  transpose2_output_y_desc.SetOriginShape(reshape_1_shape);
  transpose_2.update_input_desc_x(transpose2_input_x_desc);
  transpose_2.update_output_desc_y(transpose2_output_y_desc);

  // reshape const
  TensorDesc desc_input_size_2(x1_shape_shape, FORMAT_ND, DT_INT32);
  Tensor reshape2_const_tensor(desc_input_size_2);
  uint32_t *reshape2_const_tensor_value = new uint32_t[4];
  for (size_t dim = 0; dim < 4; dim++) {
    *(reshape2_const_tensor_value + dim) = x1_shape.GetDim(dim);
  }
  reshape2_const_tensor.SetData((uint8_t *)reshape2_const_tensor_value, 4 * sizeof(uint32_t));

  auto reshape2_const = op::Const("reshape2_const").set_attr_value(reshape2_const_tensor);
  auto reshape_2 = op::Reshape("reshape_2").set_input_x(transpose_2).set_input_shape(reshape2_const);
  ge::TensorDesc reshape2_input_x_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc reshape2_shape_desc(x1_shape_shape, FORMAT_ND, DT_INT32);
  ge::TensorDesc reshape2_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  reshape2_input_x_desc.SetOriginShape(reshape_1_shape);
  reshape2_output_y_desc.SetOriginShape(x1_shape);
  reshape_2.update_input_desc_x(reshape2_input_x_desc);
  reshape_1.update_input_desc_shape(reshape2_shape_desc);
  reshape_2.update_output_desc_y(reshape2_output_y_desc);

  ge::TensorDesc bmm3_x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor bmm3_x2_tensor(bmm3_x2_desc);
  auto bmm3_const_x2 = op::Const().set_attr_value(bmm3_x2_tensor);

  auto batchmatmul_3 = op::BatchMatMulV2("batchmatmul_3")
                           .set_input_x1(reshape_2)
                           .set_input_x2(bmm3_const_x2)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm3_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm3_input_x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm3_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  bmm3_input_x1_desc.SetOriginShape(x1_shape);
  bmm3_input_x2_desc.SetOriginShape(x2_shape);
  bmm3_output_y_desc.SetOriginShape(x1_shape);
  batchmatmul_3.update_input_desc_x1(bmm3_input_x1_desc);
  batchmatmul_3.update_input_desc_x2(bmm3_input_x2_desc);
  batchmatmul_3.update_output_desc_y(bmm3_output_y_desc);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(batchmatmul_3);
  std::vector<Operator> inputs{data_x1, const_x2, add_const, bmm2_const_x2, bmm3_const_x2};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulNonAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  EXPECT_EQ(ExecuteCorrectly(compute_graph_ptr), true);
}

TEST_F(batch_matmul_v2_non_aligned_fusion_pass_test, batchmatmul_pattern_2) {
  std::cout << "enter batch_matmul_v2_non_aligned_fusion_pass_test.batchmatmul_pattern_2" << std::endl;
  ge::Graph graph("batchmatmul_pattern_2");
  ge::Shape x1_shape({1000, 48, 324});
  ge::Shape x2_shape({324, 324});
  ge::Shape add_const_shape({324});
  ge::Shape reshape_1_shape({1000, 48, 12, 27});
  ge::Shape transpose_1_shape({1000, 12, 48, 27});
  ge::Shape transpose_2_shape({1000, 12, 27, 48});
  ge::Shape bmm3_shape({1000, 12, 48, 48});
  ge::Shape reshape_1_shape_shape({4});

  ge::TensorDesc x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  x1_desc.SetOriginShape(x1_shape);
  auto data_x1 = op::Data("data_x1");
  data_x1.update_input_desc_x(x1_desc);
  data_x1.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor x2_tensor(x2_desc);
  auto const_x2 = op::Const().set_attr_value(x2_tensor);

  auto batchmatmul_1 = op::BatchMatMulV2("batchmatmul_1")
                           .set_input_x1(data_x1)
                           .set_input_x2(const_x2)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm1_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm1_input_x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm1_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  bmm1_input_x1_desc.SetOriginShape(x1_shape);
  bmm1_input_x2_desc.SetOriginShape(x2_shape);
  bmm1_output_y_desc.SetOriginShape(x1_shape);
  batchmatmul_1.update_input_desc_x1(bmm1_input_x1_desc);
  batchmatmul_1.update_input_desc_x2(bmm1_input_x2_desc);
  batchmatmul_1.update_output_desc_y(bmm1_output_y_desc);

  ge::TensorDesc add_const_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor add_const_tensor(add_const_desc);
  auto add_const = op::Const().set_attr_value(add_const_tensor);

  auto add_1 = op::Add("add_1").set_input_x1(add_const).set_input_x2(batchmatmul_1);

  ge::TensorDesc add1_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add1_input_x2_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add1_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  add1_input_x1_desc.SetOriginShape(x1_shape);
  add1_input_x2_desc.SetOriginShape(add_const_shape);
  add1_output_y_desc.SetOriginShape(x1_shape);
  add_1.update_input_desc_x1(add1_input_x1_desc);
  add_1.update_input_desc_x2(add1_input_x2_desc);
  add_1.update_output_desc_y(add1_output_y_desc);

  // reshape const
  TensorDesc desc_input_size_1(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  Tensor reshape_const_tensor(desc_input_size_1);
  uint32_t *reshape_const_tensor_value = new uint32_t[4];
  for (size_t dim = 0; dim < 4; dim++) {
    *(reshape_const_tensor_value + dim) = reshape_1_shape.GetDim(dim);
  }
  reshape_const_tensor.SetData((uint8_t *)reshape_const_tensor_value, 4 * sizeof(uint32_t));

  auto reshape_const = op::Const("reshape_1_const").set_attr_value(reshape_const_tensor);

  auto reshape_1 = op::Reshape("reshape_1").set_input_x(add_1).set_input_shape(reshape_const);
  ge::TensorDesc reshape_input_x_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc reshape_shape_desc(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  ge::TensorDesc reshape_output_y_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  reshape_input_x_desc.SetOriginShape(x1_shape);
  reshape_output_y_desc.SetOriginShape(reshape_1_shape);
  reshape_1.update_input_desc_x(reshape_input_x_desc);
  reshape_1.update_input_desc_shape(reshape_shape_desc);
  reshape_1.update_output_desc_y(reshape_output_y_desc);

  auto transpose_1 = op::TransposeD("transpose_1").set_input_x(reshape_1).set_attr_perm({0, 2, 1, 3});
  ge::TensorDesc transpose_input_x_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc transpose_output_y_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  transpose_input_x_desc.SetOriginShape(reshape_1_shape);
  transpose_output_y_desc.SetOriginShape(transpose_1_shape);
  transpose_1.update_input_desc_x(transpose_input_x_desc);
  transpose_1.update_output_desc_y(transpose_output_y_desc);

  ge::TensorDesc x3_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  x3_desc.SetOriginShape(x1_shape);
  auto data_x3 = op::Data("data_x3");
  data_x3.update_input_desc_x(x3_desc);
  data_x3.update_output_desc_y(x3_desc);

  ge::TensorDesc x4_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor x4_tensor(x4_desc);
  auto const_x4 = op::Const().set_attr_value(x4_tensor);

  auto batchmatmul_2 = op::BatchMatMulV2("batchmatmul_2")
                           .set_input_x1(data_x3)
                           .set_input_x2(const_x4)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm2_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm2_input_x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm2_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  bmm2_input_x1_desc.SetOriginShape(x1_shape);
  bmm2_input_x2_desc.SetOriginShape(x2_shape);
  bmm2_output_y_desc.SetOriginShape(x1_shape);
  batchmatmul_2.update_input_desc_x1(bmm2_input_x1_desc);
  batchmatmul_2.update_input_desc_x2(bmm2_input_x2_desc);
  batchmatmul_2.update_output_desc_y(bmm2_output_y_desc);

  ge::TensorDesc add2_const_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor add2_const_tensor(add2_const_desc);
  auto add2_const = op::Const().set_attr_value(add2_const_tensor);

  auto add_2 = op::Add("add_2").set_input_x1(add2_const).set_input_x2(batchmatmul_2);

  ge::TensorDesc add2_input_x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add2_input_x2_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc add2_output_y_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  add2_input_x1_desc.SetOriginShape(x1_shape);
  add2_input_x2_desc.SetOriginShape(add_const_shape);
  add2_output_y_desc.SetOriginShape(x1_shape);
  add_2.update_input_desc_x1(add2_input_x1_desc);
  add_2.update_input_desc_x2(add2_input_x2_desc);
  add_2.update_output_desc_y(add2_output_y_desc);

  // reshape const
  TensorDesc desc_input_size_2(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  Tensor reshape2_const_tensor(desc_input_size_1);
  uint32_t *reshape2_const_tensor_value = new uint32_t[4];
  for (size_t dim = 0; dim < 4; dim++) {
    *(reshape2_const_tensor_value + dim) = reshape_1_shape.GetDim(dim);
  }
  reshape2_const_tensor.SetData((uint8_t *)reshape2_const_tensor_value, 4 * sizeof(uint32_t));

  auto reshape2_const = op::Const("reshape_2_const").set_attr_value(reshape2_const_tensor);

  auto reshape_2 = op::Reshape("reshape_2").set_input_x(add_2).set_input_shape(reshape2_const);
  ge::TensorDesc reshape2_input_x_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc reshape2_output_y_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc reshape2_shape_desc(reshape_1_shape_shape, FORMAT_ND, DT_INT32);
  reshape2_input_x_desc.SetOriginShape(x1_shape);
  reshape2_output_y_desc.SetOriginShape(reshape_1_shape);
  reshape_2.update_input_desc_x(reshape2_input_x_desc);
  reshape_2.update_output_desc_y(reshape2_output_y_desc);
  reshape_2.update_input_desc_shape(reshape2_shape_desc);

  auto transpose_2 = op::TransposeD("transpose_2").set_input_x(reshape_2).set_attr_perm({0, 2, 3, 1});
  ge::TensorDesc transpose2_input_x_desc(reshape_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc transpose2_output_y_desc(transpose_2_shape, FORMAT_ND, DT_FLOAT16);
  transpose2_input_x_desc.SetOriginShape(reshape_1_shape);
  transpose2_output_y_desc.SetOriginShape(transpose_2_shape);
  transpose_2.update_input_desc_x(transpose2_input_x_desc);
  transpose_2.update_output_desc_y(transpose2_output_y_desc);

  auto batchmatmul_3 = op::BatchMatMulV2("batchmatmul_3")
                           .set_input_x1(transpose_1)
                           .set_input_x2(transpose_2)
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(false);

  ge::TensorDesc bmm3_input_x1_desc(transpose_1_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm3_input_x2_desc(transpose_2_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc bmm3_output_y_desc(bmm3_shape, FORMAT_ND, DT_FLOAT16);
  bmm3_input_x1_desc.SetOriginShape(transpose_1_shape);
  bmm3_input_x2_desc.SetOriginShape(transpose_2_shape);
  bmm3_output_y_desc.SetOriginShape(bmm3_shape);
  batchmatmul_3.update_input_desc_x1(bmm3_input_x1_desc);
  batchmatmul_3.update_input_desc_x2(bmm3_input_x2_desc);
  batchmatmul_3.update_output_desc_y(bmm3_output_y_desc);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(batchmatmul_3);
  std::vector<Operator> inputs{data_x1, const_x2, add_const, data_x3, const_x4, add2_const};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulNonAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  EXPECT_EQ(ExecuteCorrectly(compute_graph_ptr), true);
}
