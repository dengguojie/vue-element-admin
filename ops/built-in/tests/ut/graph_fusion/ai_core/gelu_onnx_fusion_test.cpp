#include <algorithm>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class gelu_onnx_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "gelu_onnx_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelu_onnx_fusion_test TearDown" << std::endl;
  }
};

TEST_F(gelu_onnx_fusion_test, gelu_onnx_fusion_test1) {
  ge::Graph graph("gelu_onnx_fusion_test");
  
  std::vector<float> const_v_scales = {1.41421, 1.0, 0.5};
  std::vector<ge::op::Const> const_op_s = {};
  for (auto const_v_scale : const_v_scales) {
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&const_v_scale), sizeof(float));
    auto const_op = ge::op::Const().set_attr_value(tensor);
    const_op_s.emplace_back(const_op);
  }
  
  std::vector<int64_t> axes;
  axes.push_back(-1);
  auto data0 = op::Data().set_attr_index(0);
  auto div0 = op::RealDiv("div0").set_input_x1(data0).set_input_x2(const_op_s[0]);
  auto erf0 = op::Erf("erf0").set_input_x(div0);
  auto add0 = op::Add("add0").set_input_x1(erf0).set_input_x2(const_op_s[1]);
  auto mul0 = op::Mul("mul0").set_input_x1(data0).set_input_x2(const_op_s[2]);
  auto mul1 = op::Mul("mul1").set_input_x1(mul0).set_input_x2(add0);
  ge::TensorDesc data0_desc(ge::Shape({1, 512, 3072}), FORMAT_ND,  DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  div0.update_input_desc_x1(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{mul1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("GeluONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 512, 3072};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Gelu") {
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

TEST_F(gelu_onnx_fusion_test, gelu_onnx_fusion_test2) {
  ge::Graph graph("gelu_onnx_fusion_test");

  std::vector<float> const_v_scales = {1.41421, 1.0, 0.5};
  std::vector<ge::op::Const> const_op_s = {};
  for (auto const_v_scale : const_v_scales) {
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&const_v_scale), sizeof(float));
    auto const_op = ge::op::Const().set_attr_value(tensor);
    const_op_s.emplace_back(const_op);
  }
  std::vector<int64_t> axes;
  axes.push_back(-1);
  auto data0 = op::Data().set_attr_index(0);
  auto div0 = op::RealDiv("div0").set_input_x1(data0).set_input_x2(const_op_s[0]);
  auto erf0 = op::Erf("erf0").set_input_x(div0);
  auto add0 = op::Add("add0").set_input_x1(erf0).set_input_x2(const_op_s[1]);
  auto mul0 = op::Mul("mul0").set_input_x1(data0).set_input_x2(add0);
  auto mul1 = op::Mul("mul1").set_input_x1(mul0).set_input_x2(const_op_s[2]);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW,  DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  div0.update_input_desc_x1(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{mul1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("GeluONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Gelu") {
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

TEST_F(gelu_onnx_fusion_test, gelu_onnx_fusion_test3) {
  ge::Graph graph("gelu_onnx_fusion_test");

  std::vector<float> const_v_scales = {1.41456, 2.0, 0.5};
  std::vector<ge::op::Const> const_op_s = {};
  for (auto const_v_scale : const_v_scales) {
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&const_v_scale), sizeof(float));
    auto const_op = ge::op::Const().set_attr_value(tensor);
    const_op_s.emplace_back(const_op);
  }
  std::vector<int64_t> axes;
  axes.push_back(-1);
  auto data0 = op::Data().set_attr_index(0);
  auto div0 = op::RealDiv("div0").set_input_x1(data0).set_input_x2(const_op_s[0]);
  auto erf0 = op::Erf("erf0").set_input_x(div0);
  auto add0 = op::Add("add0").set_input_x1(erf0).set_input_x2(const_op_s[1]);
  auto mul0 = op::Mul("mul0").set_input_x1(data0).set_input_x2(add0);
  auto mul1 = op::Mul("mul1").set_input_x1(mul0).set_input_x2(const_op_s[2]);

  ge::TensorDesc data0_desc(ge::Shape({1, 3, 224, 224}), FORMAT_NCHW,  DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  div0.update_input_desc_x1(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{mul1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("GeluONNXFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{1, 3, 224, 224};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Gelu") {
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
