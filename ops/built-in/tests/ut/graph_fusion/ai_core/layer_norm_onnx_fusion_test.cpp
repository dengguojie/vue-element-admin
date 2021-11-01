#include <algorithm>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class layer_norm_onnx_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "layer_norm_onnx_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "layer_norm_onnx_fusion_test TearDown" << std::endl;
  }
};

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_1) {
  ge::Graph graph("layer_norm_onnx_fusion_test_1");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(224, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(224, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_2) {
  ge::Graph graph("layer_norm_onnx_fusion_test_2");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{div0};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_3) {
  ge::Graph graph("layer_norm_onnx_fusion_test_3");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 3.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{div0};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, false);
  EXPECT_EQ(shapeMatch, false);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_4) {
  ge::Graph graph("layer_norm_onnx_fusion_test_4");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(224, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(224, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::Div("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_5) {
  ge::Graph graph("layer_norm_onnx_fusion_test_5");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 3.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::Div("div0").set_input_x1(sub0).set_input_x2(sqrt0);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{div0};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, false);
  EXPECT_EQ(shapeMatch, false);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_6) {
  ge::Graph graph("layer_norm_onnx_fusion_test_6");

  ge::TensorDesc cast0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float cast0_data = 2.;
  ge::Tensor cast0_tensor(cast0_desc, reinterpret_cast<uint8_t*>(&cast0_data), sizeof(float));

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 3.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(224, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(224, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto cast0_const_op = op::Constant().set_attr_value(cast0_tensor);
  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  cast0_const_op.update_output_desc_y(cast0_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto cast0 = op::Cast("cast0").set_input_x(sub0);
  auto pow0 = op::Pow("pow0").set_input_x1(cast0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::Div("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, false);
  EXPECT_EQ(shapeMatch, false);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_7) {
  ge::Graph graph("layer_norm_onnx_fusion_test_7");

  ge::TensorDesc reducemean0_desc(ge::Shape({1}), FORMAT_NCHW, DT_INT32);
  int reducemean0_data = 2;
  ge::Tensor reducemean0_tensor(reducemean0_desc, reinterpret_cast<uint8_t*>(&reducemean0_data), sizeof(int));

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc reducemean1_desc(ge::Shape({1}), FORMAT_NCHW, DT_INT32);
  int reducemean1_data = 2.;
  ge::Tensor reducemean1_tensor(reducemean1_desc, reinterpret_cast<uint8_t*>(&reducemean1_data), sizeof(int));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({256}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(256, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({256}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(256, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto reducemean0_const_op = op::Constant().set_attr_value(reducemean0_tensor);
  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto reducemean1_const_op = op::Constant().set_attr_value(reducemean1_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  reducemean0_const_op.update_output_desc_y(reducemean0_desc);
  pow0_const_op.update_output_desc_y(pow0_desc);
  reducemean1_const_op.update_output_desc_y(reducemean1_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMean("mean0").set_input_x(data0).set_input_axes(reducemean0_const_op).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMean("mean1").set_input_x(pow0).set_input_axes(reducemean1_const_op).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 7, 256}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  mean0.update_input_desc_axes(reducemean0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 7, 256};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}


TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_8) {
  ge::Graph graph("layer_norm_onnx_fusion_test_8");

  ge::TensorDesc reducemean0_desc(ge::Shape({1}), FORMAT_NCHW, DT_INT64);
  int64_t reducemean0_data = 2;
  ge::Tensor reducemean0_tensor(reducemean0_desc, reinterpret_cast<uint8_t*>(&reducemean0_data), sizeof(int64_t));

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc reducemean1_desc(ge::Shape({1}), FORMAT_NCHW, DT_INT64);
  int64_t reducemean1_data = 2;
  ge::Tensor reducemean1_tensor(reducemean1_desc, reinterpret_cast<uint8_t*>(&reducemean1_data), sizeof(int64_t));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({256}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(256, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({256}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(256, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto reducemean0_const_op = op::Constant().set_attr_value(reducemean0_tensor);
  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto reducemean1_const_op = op::Constant().set_attr_value(reducemean1_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  reducemean0_const_op.update_output_desc_y(reducemean0_desc);
  pow0_const_op.update_output_desc_y(pow0_desc);
  reducemean1_const_op.update_output_desc_y(reducemean1_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMean("mean0").set_input_x(data0).set_input_axes(reducemean0_const_op).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMean("mean1").set_input_x(pow0).set_input_axes(reducemean1_const_op).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 7, 256}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  mean0.update_input_desc_axes(reducemean0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 7, 256};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}


TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_9) {
  ge::Graph graph("layer_norm_onnx_fusion_test_9");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 1.0;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  ge::TensorDesc mul0_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  std::vector<int> mul0_data(224, 1);
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(mul0_data.data()), sizeof(int) * mul0_size);

  ge::TensorDesc add1_desc(ge::Shape({224}), FORMAT_NCHW, DT_FLOAT);
  int64_t add1_size = add1_desc.GetShape().GetShapeSize();
  std::vector<int> add1_data(224, 1);
  ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), sizeof(int) * add1_size);

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  add1_const_op.update_output_desc_y(add1_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);
  auto mul0 = op::Mul("mul0").set_input_x1(div0).set_input_x2(mul0_const_op);
  auto add1 = op::Add("add1").set_input_x1(mul0).set_input_x2(add1_const_op);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, false);
  EXPECT_EQ(shapeMatch, false);
}

TEST_F(layer_norm_onnx_fusion_test, layer_norm_onnx_fusion_test_10) {
  ge::Graph graph("layer_norm_onnx_fusion_test_10");

  ge::TensorDesc pow0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float pow0_data = 2.;
  ge::Tensor pow0_tensor(pow0_desc, reinterpret_cast<uint8_t*>(&pow0_data), sizeof(float));

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT);
  float add0_data = 0.001;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), sizeof(float));

  auto pow0_const_op = op::Constant().set_attr_value(pow0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  pow0_const_op.update_output_desc_y(pow0_desc);
  add0_const_op.update_output_desc_y(add0_desc);

  std::vector<int64_t> axes = {-1};
  auto data0 = op::Data().set_attr_index(0);
  auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto sub0 = op::Sub("sub0").set_input_x1(data0).set_input_x2(mean0);
  auto pow0 = op::Pow("pow0").set_input_x1(sub0).set_input_x2(pow0_const_op);
  auto mean1 = op::ReduceMeanD("mean1").set_input_x(pow0).set_attr_axes(axes).set_attr_keep_dims(true);
  auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
  auto sqrt0 = op::Sqrt("sqrt0").set_input_x(add0);
  auto div0 = op::RealDiv("div0").set_input_x1(sub0).set_input_x2(sqrt0);

  ge::TensorDesc data0_desc(ge::Shape({-1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  mean0.update_input_desc_x(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{div0};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{-1, 3, 224, 224};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "LayerNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, false);
  EXPECT_EQ(shapeMatch, false);
}