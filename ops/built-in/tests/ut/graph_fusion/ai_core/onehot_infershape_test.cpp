#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class one_hot_infershapeone_hot_infershape_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "one_hot_infershapeone_hot_infershape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "one_hot_infershapeone_hot_infershape TearDown" << std::endl;
  }
};
// REG_OP(OneHot)
//     .INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
//     .INPUT(depth, TensorType({DT_INT32}))
//     .INPUT(on_value, TensorType::BasicType())
//     .INPUT(off_value, TensorType::BasicType())
//     .OUTPUT(y, TensorType::BasicType())
//     .ATTR(axis, Int, -1)
//     .OP_END_FACTORY_REG(OneHot)
TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_1) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_1");
  // input x info
  auto input_x_shape = vector<int64_t>({-1, -1, -1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  uint32_t depth = 101;
  int64_t axis = -1;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-1, -1, -1, 5, int64_t(depth)};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {2, 3}, {5, 5}, {int64_t(depth), int64_t(depth)}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[1]{depth};
  depth_tensor.SetData((uint8_t *) depth_tensor_value, sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);

  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_const_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_const_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
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

TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_2) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_2");
  // input x info
  auto input_x_shape = vector<int64_t>({32});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  uint32_t depth = 1001;
  int64_t axis = -1;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {32, int64_t(depth)};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {2, 3}, {5, 5}, {1, int64_t(depth)}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[1]{depth};
  depth_tensor.SetData((uint8_t *) depth_tensor_value, sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);

  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_const_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_const_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      // EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);
  delete[] depth_tensor_value;
}


TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_3) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_3");
  // input x info
  auto input_x_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  uint32_t depth = 1001;
  int64_t axis = -1;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-2};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {2, 3}, {2, 3}, {5, 5}, {1, int64_t(depth)}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[1]{depth};
  depth_tensor.SetData((uint8_t *) depth_tensor_value, sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);

  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_const_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_const_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t,int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      // EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);
  delete[] depth_tensor_value;
}

TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_4) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_4");
  // input x info
  auto input_x_shape = vector<int64_t>({32, 32, 32});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{2, 3}, {2, 4}, {5, 5}};
  uint32_t depth = 1001;
  int64_t axis = 0;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-1, 32, 32, 32};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {32, 32}, {32, 32}, {32, 32}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  // desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  auto depth_op = op::Data("depth");
  depth_op.update_input_desc_x(desc_input_size_1);
  depth_op.update_output_desc_y(desc_input_size_1);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);

  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
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
}


TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_5) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_5");
  // input x info
  auto input_x_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 3}, {1, 4}, {1, 5}};
  uint32_t depth = 1001;
  int64_t axis = 2;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-1, -1, 1001, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 3}, {1, 4}, {1001, 1001}, {1, 5}};


  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t *depth_tensor_value = new uint32_t[1]{depth};
  depth_tensor.SetData((uint8_t *) depth_tensor_value, sizeof(uint32_t));
  auto depth_const_op = op::Const("depth")
                       .set_attr_value(depth_tensor);

  auto depth_op = op::Data("depth");
  depth_op.update_input_desc_x(desc_input_size_1);
  depth_op.update_output_desc_y(desc_input_size_1);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);



  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_const_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_const_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
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

TEST_F(one_hot_infershapeone_hot_infershape_pass_test, one_hot_infershapeone_hot_infershape_pass_test_6) {
  ge::Graph graph("one_hot_infershapeone_hot_infershape_pass_test_6");
  // input x info
  auto input_x_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, 3}, {1, 4}, {1, 5}};
  uint32_t depth = 1001;
  int64_t axis = 2;
  auto dtype = DT_FLOAT;
  // expect info
  std::vector<int64_t> expected_shape = {-1, -1, 1001, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, 3}, {1, 4}, {1001, 1001}, {1, 5}};

  // input x desc
  TensorDesc desc_data(ge::Shape(input_x_shape), FORMAT_NCHW, DT_INT32);
  desc_data.SetShapeRange(range_x1);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // depth scaler const
  vector<int64_t> depth_dims;
  TensorDesc desc_input_size_1(ge::Shape(depth_dims), FORMAT_ND, DT_INT32);
  Tensor depth_tensor(desc_input_size_1);
  uint32_t* depth_tensor_value = new uint32_t[1]{depth};
  depth_tensor.SetData((uint8_t*)depth_tensor_value, sizeof(uint32_t));
  auto depth_const_op = op::Const("depth").set_attr_value(depth_tensor);

  auto depth_op = op::Data("depth");
  depth_op.update_input_desc_x(desc_input_size_1);
  depth_op.update_output_desc_y(desc_input_size_1);
  // on_value / off_value tensor
  auto value_data = vector<int64_t>({});
  TensorDesc value_desc(ge::Shape(value_data), FORMAT_NCHW, dtype);
  auto value_op = op::Data("value_data");
  value_op.update_input_desc_x(value_desc);
  value_op.update_output_desc_y(value_desc);

  // new op
  auto test_op = op::OneHot("OneHot");
  test_op.set_input_x(data);
  test_op.set_input_depth(depth_const_op);
  test_op.set_input_on_value(value_op);
  test_op.set_input_off_value(value_op);
  test_op.SetAttr("axis", axis);
  std::vector<Operator> inputs{data, depth_const_op, value_op};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHot") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      std::vector<std::pair<int64_t, int64_t>> output_range;
      output_desc.GetShapeRange(output_range);
      EXPECT_EQ(output_range, expected_range);
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);
  delete[] depth_tensor_value;

  fe::FusionPassTestUtils::RunGraphFusionPass("OneHotFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool ret = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "OneHotD") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, true);
}
