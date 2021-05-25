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
  
  ge::TensorDesc div0_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
  int64_t div0_size = div0_desc.GetShape().GetShapeSize();
  float div0_data = 1.4142099618911743;
  ge::Tensor div0_tensor(div0_desc, reinterpret_cast<uint8_t*>(&div0_data), div0_size);

  ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
  int64_t add0_size = add0_desc.GetShape().GetShapeSize();
  float add0_data = 1.0;
  ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), add0_size);
  
  ge::TensorDesc mul0_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
  int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
  float mul0_data = 0.5;
  ge::Tensor mul0_tensor(mul0_desc, reinterpret_cast<uint8_t*>(&mul0_data), mul0_size);

  auto div0_const_op = op::Constant().set_attr_value(div0_tensor);
  auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
  auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
  div0_const_op.update_output_desc_y(div0_desc);
  add0_const_op.update_output_desc_y(add0_desc);
  mul0_const_op.update_output_desc_y(mul0_desc);
  
  std::vector<int64_t> axes;
  axes.push_back(-1);
  auto data0 = op::Data().set_attr_index(0);
  auto div0 = op::RealDiv("div0").set_input_x1(data0).set_input_x2(div0_const_op);
  auto erf0 = op::Erf("erf0").set_input_x(div0);
  auto add0 = op::Add("add0").set_input_x1(erf0).set_input_x2(add0_const_op);
  auto mul0 = op::Mul("mul0").set_input_x1(data0).set_input_x2(mul0_const_op);
  auto mul1 = op::Mul("mul1").set_input_x1(mul0).set_input_x2(add0);
  ge::TensorDesc data0_desc(ge::Shape({1, 512, 3072}), FORMAT_ND,  DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  div0.update_input_desc_x1(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{mul1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
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