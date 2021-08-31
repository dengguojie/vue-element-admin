#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pad_v3_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "pad_v3_fusion_pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pad_v3_fusion_pass TearDown" << std::endl;
  }
};

TEST_F(pad_v3_fusion_pass_test, pad_v3_fusion_pass_test_1) {
  ge::Graph graph("pad_v3_fusion_pass_test_1");
  auto shape_data = vector<int64_t>({1, 3, 4, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT);

  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  auto multiples_shape = ge::Shape({8});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor paddings_tensor(desc_input_size_1);
  uint32_t *paddings_tensor_value = new uint32_t[8]{0, 1, 0, 2, 0, 3, 0, 4};
  paddings_tensor.SetData((uint8_t *) paddings_tensor_value, 8 * sizeof(uint32_t));

  auto pad_v3_paddings = op::Const("paddings")
                         .set_attr_value(paddings_tensor);
  // pad_v3d op
  auto pad_v3 = op::PadV3("pad_v3");
  pad_v3.set_input_x(data);
  pad_v3.set_input_paddings(pad_v3_paddings);

  TensorDesc pad_v3_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_input_desc_paddings(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  pad_v3.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3.update_output_desc_y(pad_v3_output_desc_y);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{pad_v3};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("PadV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_after");

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3D") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] paddings_tensor_value;

}

TEST_F(pad_v3_fusion_pass_test, pad_v3_fusion_pass_test_2) {
  ge::Graph graph("pad_v3_fusion_pass_test_2");
  auto shape_data = vector<int64_t>({1, 3, 4, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT);

  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  auto multiples_shape = ge::Shape({8});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT64);
  Tensor paddings_tensor(desc_input_size_1);
  uint32_t *paddings_tensor_value = new uint32_t[8]{0, 1, 0, 2, 0, 3, 0, 4};
  paddings_tensor.SetData((uint8_t *) paddings_tensor_value, 8 * sizeof(uint32_t));
  auto const_v_shape = ge::Shape({1});
  TensorDesc desc_input_size_2(const_v_shape, FORMAT_ND, DT_FLOAT);
  Tensor const_v_tensor(desc_input_size_2);
  float *const_v_tensor_value = new float[1]{0};
  const_v_tensor.SetData((uint8_t *) const_v_tensor_value, 1 * sizeof(float));

  auto pad_v3_paddings = op::Const("paddings")
                         .set_attr_value(paddings_tensor);
  auto pad_v3_const_v = op::Const("constant_values")
                         .set_attr_value(const_v_tensor);
  // pad_v3d op
  auto pad_v3 = op::PadV3("pad_v3");
  pad_v3.set_input_x(data);
  pad_v3.set_input_paddings(pad_v3_paddings);
  pad_v3.set_input_constant_values(pad_v3_const_v);

  TensorDesc pad_v3_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_input_desc_paddings(ge::Shape(), FORMAT_NCHW, DT_INT64);
  TensorDesc pad_v3_input_desc_constant_values(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  pad_v3.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3.update_input_desc_constant_values(pad_v3_input_desc_constant_values);
  pad_v3.update_output_desc_y(pad_v3_output_desc_y);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{pad_v3};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("PadV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_after");

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] paddings_tensor_value;

}

TEST_F(pad_v3_fusion_pass_test, pad_v3_fusion_pass_test_3) {
  // unknow shape
  ge::Graph graph("pad_v3_fusion_pass_test_3");
  auto shape_data = vector<int64_t>({1, -1, 4, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT);

  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  auto multiples_shape = ge::Shape({8});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  Tensor paddings_tensor(desc_input_size_1);
  uint32_t *paddings_tensor_value = new uint32_t[8]{0, 1, 0, 2, 0, 3, 0, 4};
  paddings_tensor.SetData((uint8_t *) paddings_tensor_value, 8 * sizeof(uint32_t));

  auto pad_v3_paddings = op::Const("paddings")
                         .set_attr_value(paddings_tensor);
  // pad_v3d op
  auto pad_v3 = op::PadV3("pad_v3");
  pad_v3.set_input_x(data);
  pad_v3.set_input_paddings(pad_v3_paddings);

  TensorDesc pad_v3_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_input_desc_paddings(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  vector<int64_t> y_dims = {-1, -1, -1};
  ge::Shape shape_y(y_dims);
  TensorDesc pad_v3_output_desc_y(shape_y, FORMAT_NCHW, DT_FLOAT);
  pad_v3.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3.update_output_desc_y(pad_v3_output_desc_y);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{pad_v3};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("PadV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_after");

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] paddings_tensor_value;
}

TEST_F(pad_v3_fusion_pass_test, pad_v3_fusion_pass_test_4) {
  // no const input
  ge::Graph graph("pad_v3_fusion_pass_test_4");
  auto shape_data = vector<int64_t>({1, 3, 4, -1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT);

  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  auto shape_padding = ge::Shape({8});
  TensorDesc desc_padding(shape_padding, FORMAT_ND, DT_INT32);
  auto pad_v3_paddings = op::Data("paddings");
  pad_v3_paddings.update_input_desc_x(desc_padding);
  pad_v3_paddings.update_output_desc_y(desc_padding);
  auto shape_const_v = ge::Shape({1});
  TensorDesc desc_const_v(shape_const_v, FORMAT_ND, DT_FLOAT);
  auto pad_v3_constant_v = op::Data("constant_values");
  pad_v3_constant_v.update_input_desc_x(desc_const_v);
  pad_v3_constant_v.update_output_desc_y(desc_const_v);
  
  // pad_v3d op
  auto pad_v3 = op::PadV3("pad_v3");
  pad_v3.set_input_x(data);
  pad_v3.set_input_paddings(pad_v3_paddings);
  pad_v3.set_input_constant_values(pad_v3_constant_v);

  TensorDesc pad_v3_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_input_desc_paddings(ge::Shape(), FORMAT_NCHW, DT_INT32);
  TensorDesc pad_v3_input_desc_constant_values(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  pad_v3.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3.update_input_desc_constant_values(pad_v3_input_desc_constant_values);
  pad_v3.update_output_desc_y(pad_v3_output_desc_y);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{pad_v3};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("PadV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_after");

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
}

TEST_F(pad_v3_fusion_pass_test, pad_v3_fusion_pass_test_5) {
  ge::Graph graph("pad_v3_fusion_pass_test_5");
  auto shape_data = vector<int64_t>({1, 3, 4, 5});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT);

  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  auto multiples_shape = ge::Shape({8});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT64);
  Tensor paddings_tensor(desc_input_size_1);
  uint32_t *paddings_tensor_value = new uint32_t[8]{0, 1, 0, 2, 0, 3, 0, 4};
  paddings_tensor.SetData((uint8_t *) paddings_tensor_value, 8 * sizeof(uint32_t));
  auto const_v_shape = ge::Shape({1});
  TensorDesc desc_input_size_2(const_v_shape, FORMAT_ND, DT_FLOAT);
  Tensor const_v_tensor(desc_input_size_2);
  float *const_v_tensor_value = new float[1]{0};
  const_v_tensor.SetData((uint8_t *) const_v_tensor_value, 1 * sizeof(float));

  auto pad_v3_paddings = op::Const("paddings")
                         .set_attr_value(paddings_tensor);
  auto pad_v3_const_v = op::Const("constant_values")
                         .set_attr_value(const_v_tensor);
  // pad_v3d op
  auto pad_v3 = op::PadV3("pad_v3_0");
  pad_v3.set_input_x(data);
  pad_v3.set_input_paddings(pad_v3_paddings);
  pad_v3.set_input_constant_values(pad_v3_const_v);
  
  auto pad_v3_node2 = op::PadV3("pad_v3_1");
  pad_v3_node2.set_input_x(pad_v3);
  pad_v3_node2.set_input_paddings(pad_v3_paddings);
  pad_v3_node2.set_input_constant_values(pad_v3_const_v);

  TensorDesc pad_v3_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_input_desc_paddings(ge::Shape(), FORMAT_NCHW, DT_INT64);
  TensorDesc pad_v3_input_desc_constant_values(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  TensorDesc pad_v3_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
  pad_v3.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3.update_input_desc_constant_values(pad_v3_input_desc_constant_values);
  pad_v3.update_output_desc_y(pad_v3_output_desc_y);

  pad_v3_node2.update_input_desc_x(pad_v3_input_desc_x);
  pad_v3_node2.update_input_desc_paddings(pad_v3_input_desc_paddings);
  pad_v3_node2.update_input_desc_constant_values(pad_v3_input_desc_constant_values);
  pad_v3_node2.update_output_desc_y(pad_v3_output_desc_y);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{pad_v3_node2};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("PadV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "pad_v3_fusion_test_1_after");

  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] paddings_tensor_value;

}
