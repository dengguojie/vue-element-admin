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

class batchtospacend_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "batchtospacend SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batchtospacend TearDown" << std::endl;
  }
};

// REG_OP(BatchToSpaceND)
//     .INPUT(x, TensorType::BasicType())
//     .INPUT(block_shape, TensorType::IndexNumberType())
//     .INPUT(crops, TensorType::IndexNumberType())
//     .OUTPUT(y, TensorType::BasicType())
//     .OP_END_FACTORY_REG(BatchToSpaceND)

TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_1) {
  ge::Graph graph("batchtospacend_fusion_pass_test_1");
  auto shape_data = vector<int64_t>({4, 1, 1, 1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({2});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_block_tensor(desc_input_size_1);
  uint32_t *const_block_tensor_value = new uint32_t[2]{2, 2};
  const_block_tensor.SetData((uint8_t *) const_block_tensor_value, 2 * sizeof(uint32_t));
  auto const_block = op::Const("const_block").set_attr_value(const_block_tensor);
  delete[] const_block_tensor_value;

  // const crops shape
  auto crops_shape = ge::Shape({2, 2});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_crops_tensor(desc_input_size_crops);
  uint32_t *const_crops_tensor_value = new uint32_t[4]{0, 0, 0, 0};
  const_crops_tensor.SetData((uint8_t *) const_crops_tensor_value, 4 * sizeof(uint32_t));
  auto const_crops = op::Const("const_crops").set_attr_value(const_crops_tensor);
  delete[] const_crops_tensor_value;

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {1, 2, 2, 1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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

TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_2) {
  ge::Graph graph("batchtospacend_fusion_pass_test_2");
  auto shape_data = vector<int64_t>({8, 1, 3, 1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({2});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_block_tensor(desc_input_size_1);
  uint32_t *const_block_tensor_value = new uint32_t[2]{2, 2};
  const_block_tensor.SetData((uint8_t *) const_block_tensor_value, 2 * sizeof(uint32_t));
  auto const_block = op::Const("const_block").set_attr_value(const_block_tensor);
  delete[] const_block_tensor_value;

  // const crops shape
  auto crops_shape = ge::Shape({2, 2});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_crops_tensor(desc_input_size_crops);
  uint32_t *const_crops_tensor_value = new uint32_t[4]{0, 0, 2, 0};
  const_crops_tensor.SetData((uint8_t *) const_crops_tensor_value, 4 * sizeof(uint32_t));
  auto const_crops = op::Const("const_crops").set_attr_value(const_crops_tensor);
  delete[] const_crops_tensor_value;

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {2, 2, 4, 1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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


TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_3) {
  ge::Graph graph("batchtospacend_fusion_pass_test_3");
  auto shape_data = vector<int64_t>({8, -1, -1, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{8, 8}, {2, 3}, {2, 3}, {5, 5}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({2});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // const crops shape
  auto crops_shape = ge::Shape({2, 2});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_crops_tensor(desc_input_size_crops);
  uint32_t *const_crops_tensor_value = new uint32_t[4]{0, 0, 2, 0};
  const_crops_tensor.SetData((uint8_t *) const_crops_tensor_value, 4 * sizeof(uint32_t));
  auto const_crops = op::Const("const_crops").set_attr_value(const_crops_tensor);
  delete[] const_crops_tensor_value;

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 8}, {2, 24}, {1, 22}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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

TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_4) {
  ge::Graph graph("batchtospacend_fusion_pass_test_4");
  auto shape_data = vector<int64_t>({8, -1, -1, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{8, 8}, {2, 3}, {2, 3}, {5, 5}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({2});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // const crops shape
  auto crops_shape = ge::Shape({2, 2});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  auto const_crops = op::Data("const_crops");
  const_crops.update_input_desc_x(desc_input_size_crops);
  const_crops.update_output_desc_y(desc_input_size_crops);

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 8}, {1, 24}, {1, 24}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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

TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_5) {
  ge::Graph graph("batchtospacend_fusion_pass_test_5");
  auto shape_data = vector<int64_t>({8, -1, -1, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{8, 8}, {2, 3}, {2, 3}, {5, 5}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({2});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_block_tensor(desc_input_size_1);
  uint32_t *const_block_tensor_value = new uint32_t[2]{2, 2};
  const_block_tensor.SetData((uint8_t *) const_block_tensor_value, 2 * sizeof(uint32_t));
  auto const_block = op::Const("const_block").set_attr_value(const_block_tensor);
  delete[] const_block_tensor_value;

  // const crops shape
  auto crops_shape = ge::Shape({2, 2});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor const_crops_tensor(desc_input_size_crops);
  uint32_t *const_crops_tensor_value = new uint32_t[4]{1, 0, 1, 0};
  const_crops_tensor.SetData((uint8_t *) const_crops_tensor_value, 4 * sizeof(uint32_t));
  auto const_crops = op::Const("const_crops").set_attr_value(const_crops_tensor);
  delete[] const_crops_tensor_value;

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {2, -1, -1, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{2, 2}, {3, 5}, {3, 5}, {5, 5}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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

TEST_F(batchtospacend_fusion_pass_test, batchtospacend_fusion_pass_test_6) {
  ge::Graph graph("batchtospacend_fusion_pass_test_6");
  auto shape_data = vector<int64_t>({-1, -1, -1, -1, -1, -1, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NHWC, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 80},{1, 80}, {1, 80}, {1, 80}, {1, 80}, {1, 80}, {1, 80}};
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // const block shape
  auto multiples_shape = ge::Shape({-1});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  std::vector<std::pair<int64_t,int64_t>> multiples_range = {{2, 10}};
  desc_input_size_1.SetShapeRange(multiples_range);
  auto const_block = op::Data("const_block");
  const_block.update_input_desc_x(desc_input_size_1);
  const_block.update_output_desc_y(desc_input_size_1);

  // const crops shape
  auto crops_shape = ge::Shape({-1, -1});
  TensorDesc desc_input_size_crops(crops_shape, FORMAT_ND, DT_INT32);
  std::vector<std::pair<int64_t,int64_t>> crops_range = {{1, 10}, {2, 5}};
  desc_input_size_crops.SetShapeRange(crops_range);
  auto const_crops = op::Data("const_crops");
  const_crops.update_input_desc_x(desc_input_size_crops);
  const_crops.update_output_desc_y(desc_input_size_crops);

  // new op
  auto testOp = op::BatchToSpaceND("BatchToSpaceND");
  testOp.set_input_x(data);
  testOp.set_input_block_shape(const_block);
  testOp.set_input_crops(const_crops);
  std::vector<Operator> inputs{data, const_block, const_crops};
  std::vector<Operator> outputs{testOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 80},{1, 6400}, {1, 6400}, {1, 6400}, {1, 6400}, {1, 6400}, {1, 6400}};

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceND") {
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
