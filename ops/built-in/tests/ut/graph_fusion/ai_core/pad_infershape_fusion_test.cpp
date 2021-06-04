#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pad_infershapepad_infershape_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "pad_infershapepad_infershape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pad_infershapepad_infershape TearDown" << std::endl;
  }
};

// REG_OP(Pad)
//     .INPUT(x, TensorType::BasicType())
//     .INPUT(paddings, TensorType::IndexNumberType())
//     .OUTPUT(y, TensorType::BasicType())
//     .OP_END_FACTORY_REG(Pad)
TEST_F(pad_infershapepad_infershape_pass_test, pad_infershapepad_infershape_pass_test_1) {
  ge::Graph graph("pad_infershapepad_infershape_pass_test_1");
  // input x info
  auto input_x_shape = vector<int64_t>({3, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}};
  vector<uint32_t> paddings = {1, 1, 2, 2};
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {5, 9};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, dtype);
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // paddings scaler const
  vector<int64_t> depth_dims = {2, 2};
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[input_x_shape.size() * 2];
  for (size_t dim = 0; dim < input_x_shape.size() * 2; dim++) {
    *(depth_tensor_value + dim) = paddings[dim];
  }
  depth_tensor.SetData((uint8_t *) depth_tensor_value, input_x_shape.size() * 2 * sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);
  // new op
  auto test_op = op::Pad("Pad");
  test_op.set_input_x(data);
  test_op.set_input_paddings(depth_const_op);
  std::vector<Operator> inputs{data, depth_const_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pad") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);
  delete[] depth_tensor_value;
}

TEST_F(pad_infershapepad_infershape_pass_test, pad_infershapepad_infershape_pass_test_2) {
  ge::Graph graph("pad_infershapepad_infershape_pass_test_2");
  // input x info
  auto input_x_shape = vector<int64_t>({-1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {1, 50}};
  vector<uint32_t> paddings = {1, 1, 2, 2};
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{3, -1}, {5, 54}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, dtype);
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // paddings scaler const
  vector<int64_t> depth_dims = {input_x_shape.size(), input_x_shape.size()};
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[input_x_shape.size() * 2];
  for (size_t dim = 0; dim < input_x_shape.size()*2; dim++) {
    *(depth_tensor_value + dim) = paddings[dim];
  }
  depth_tensor.SetData((uint8_t *) depth_tensor_value, input_x_shape.size() * 2 * sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);
  // new op
  auto test_op = op::Pad("Pad");
  test_op.set_input_x(data);
  test_op.set_input_paddings(depth_const_op);
  std::vector<Operator> inputs{data, depth_const_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("PadFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pad") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);
  delete[] depth_tensor_value;
}

