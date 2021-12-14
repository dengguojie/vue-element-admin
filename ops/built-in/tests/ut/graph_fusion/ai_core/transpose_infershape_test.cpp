#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class Transpose_infer_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Transpose SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Transpose TearDown" << std::endl;
  }
};

// REG_OP(Transpose)
//     .INPUT(x, TensorType::BasicType())
//     .INPUT(perm, TensorType::IndexNumberType())
//     .OUTPUT(y, TensorType::BasicType())
//     .OP_END_FACTORY_REG(Transpose)

TEST_F(Transpose_infer_pass_test, Transpose_infer_pass_test_1) {
  ge::Graph graph("Transpose_infer_pass_test_1");
  auto shape_data = vector<int64_t>({1, 2, 3, 4});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor const_block_tensor(desc_input_size_1);
  uint32_t *const_block_tensor_value = new uint32_t[4]{3, 0, 2, 1};
  const_block_tensor.SetData((uint8_t *) const_block_tensor_value, 4 * sizeof(uint32_t));
  auto const_block = op::Const("const_block").set_attr_value(const_block_tensor);
  delete[] const_block_tensor_value;

  // new op
  auto testOp = op::Transpose("Transpose");
  testOp.set_input_x(data);
  testOp.set_input_perm(const_block);
  std::vector<Operator> inputs{data, const_block};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {4, 1, 3, 2};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      // EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(Transpose_infer_pass_test, Transpose_infer_pass_test_2) {
  ge::Graph graph("Transpose_infer_pass_test_2");
  auto shape_data = vector<int64_t>({-1, -1, -1, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 50}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor const_block_tensor(desc_input_size_1);
  uint32_t *const_block_tensor_value = new uint32_t[4]{3, 0, 2, 1};
  const_block_tensor.SetData((uint8_t *) const_block_tensor_value, 4 * sizeof(uint32_t));
  auto const_block = op::Const("const_block").set_attr_value(const_block_tensor);
  delete[] const_block_tensor_value;

  // new op
  auto testOp = op::Transpose("Transpose");
  testOp.set_input_x(data);
  testOp.set_input_perm(const_block);
  std::vector<Operator> inputs{data, const_block};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{5, 50}, {1, -1}, {2, 3}, {2, 3}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(Transpose_infer_pass_test, Transpose_infer_pass_test_3) {
  ge::Graph graph("Transpose_infer_pass_test_3");
  auto shape_data = vector<int64_t>({-1, -1, -1, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{15, -1}, {20, 3}, {20, 3}, {5, 50}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // new op
  auto testOp = op::Transpose("Transpose");
  testOp.set_input_x(data);
  testOp.set_input_perm(const_block);
  std::vector<Operator> inputs{data, const_block};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{5, -1}, {5, -1}, {5, -1}, {5, -1}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(Transpose_infer_pass_test, Transpose_infer_pass_test_4) {
  ge::Graph graph("Transpose_infer_pass_test_4");
  auto shape_data = vector<int64_t>({-2});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{15, -1}, {20, 3}, {20, 3}, {5, 50}};
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // new op
  auto testOp = op::Transpose("Transpose");
  testOp.set_input_x(data);
  testOp.set_input_perm(const_block);
  std::vector<Operator> inputs{data, const_block};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{0, -1}, {0, -1}, {0, -1}, {0, -1}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(Transpose_infer_pass_test, Transpose_infer_pass_test_5) {
  ge::Graph graph("Transpose_infer_pass_test_5");
  auto shape_data = vector<int64_t>({-2});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{15, -1}, {20, 3}, {20, 3}, {5, 50}};
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // new op
  auto testOp = op::Transpose("Transpose");
  testOp.set_input_x(data);
  testOp.set_input_perm(const_block);
  std::vector<Operator> inputs{data, const_block};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{0, -1}, {0, -1}, {0, -1}, {0, -1}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      findD = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
    }
  }
  EXPECT_EQ(findD, true);
}
