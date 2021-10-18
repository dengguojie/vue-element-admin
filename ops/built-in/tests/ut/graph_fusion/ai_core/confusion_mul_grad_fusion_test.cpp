#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class confusion_mul_grad_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_add SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(confusion_mul_grad_fusion_test, confusion_mul_grad_fusion_test_1) {
  auto graph = std::make_shared<ge::ComputeGraph>("confusion_mul_grad_fusion_test_1");

  ge::GeShape bias_shape({64});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
  bias_desc.SetOriginDataType(ge::DT_FLOAT);
  bias_desc.SetOriginShape(bias_shape);

  OpDescPtr op_desc_input1 = std::make_shared<OpDesc>("other1", "Other");
  OpDescPtr op_desc_input2 = std::make_shared<OpDesc>("other2", "Other");
  OpDescPtr op_desc_input3 = std::make_shared<OpDesc>("other3", "Other");
  OpDescPtr op_desc_mul = std::make_shared<OpDesc>("mul", "Mul");
  OpDescPtr op_desc_mul1 = std::make_shared<OpDesc>("loss/gradMul", "Mul");
  OpDescPtr op_desc_reducesumD = std::make_shared<OpDesc>("reducesumD", "ReduceSumD");
  OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

  op_desc_input1->AddOutputDesc(bias_desc);
  op_desc_input2->AddOutputDesc(bias_desc);
  op_desc_input2->AddOutputDesc(bias_desc);
  op_desc_input3->AddOutputDesc(bias_desc);

  op_desc_mul->AddInputDesc(bias_desc);
  op_desc_mul->AddInputDesc(bias_desc);
  op_desc_mul->AddOutputDesc(bias_desc);
  op_desc_mul1->AddInputDesc(bias_desc);
  op_desc_mul1->AddInputDesc(bias_desc);
  op_desc_mul1->AddOutputDesc(bias_desc);

  op_desc_reducesumD->AddInputDesc(bias_desc);
  op_desc_reducesumD->AddOutputDesc(bias_desc);

  op_desc_output->AddInputDesc(bias_desc);
  op_desc_output->AddInputDesc(bias_desc);

  NodePtr node_input1 = graph->AddNode(op_desc_input1);
  NodePtr node_input2 = graph->AddNode(op_desc_input1);
  NodePtr node_input3 = graph->AddNode(op_desc_input1);
  NodePtr node_mul = graph->AddNode(op_desc_mul);
  NodePtr node_mul1 = graph->AddNode(op_desc_mul1);
  NodePtr node_reducesumD = graph->AddNode(op_desc_reducesumD);
  NodePtr node_netoutput = graph->AddNode(op_desc_output);

  ge::AttrUtils::SetBool(node_reducesumD->GetOpDesc(), "keep_dims", true);
  ge::AttrUtils::SetListInt(node_reducesumD->GetOpDesc(), "axes", {1});

  GraphUtils::AddEdge(node_input1->GetOutDataAnchor(0), node_mul->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_input2->GetOutDataAnchor(0), node_mul1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_input2->GetOutDataAnchor(0), node_mul->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_input3->GetOutDataAnchor(0), node_mul->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_mul->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_mul1->GetOutDataAnchor(0), node_reducesumD->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reducesumD->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

  fe::FusionPassTestUtils::InferShapeAndType(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("MulGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  bool ret = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "ConfusionMulGrad") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, true);
}
TEST_F(confusion_mul_grad_fusion_test, confusion_mul_grad_fusion_test_2) {
  ge::Graph graph("confusion_mul_grad_fusion_test_2");

  auto mul_input_x_data1 = op::Data("mul_input_x_data1");
  std::vector<int64_t> dims_x{16, 16};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
  mul_input_x_data1.update_input_desc_x(tensorDescX);
  mul_input_x_data1.update_output_desc_y(tensorDescX);

  auto mul_input_x_data2 = op::Data("mul_input_x_data2");
  mul_input_x_data2.update_input_desc_x(tensorDescX);
  mul_input_x_data2.update_output_desc_y(tensorDescX);

  auto mul_input_x_data3 = op::Data("mul_input_x_data3");
  mul_input_x_data3.update_input_desc_x(tensorDescX);
  mul_input_x_data3.update_output_desc_y(tensorDescX);

  auto mul1_op = op::Mul("mul1_op");
  mul1_op.set_input_x1(mul_input_x_data3).set_input_x2(mul_input_x_data2);

  auto mul_op = op::Mul("mul_op");
  mul_op.set_input_x1(mul_input_x_data1).set_input_x2(mul_input_x_data2);

  auto sum0 = op::ReduceSumD("sum0");
  sum0.set_input_x(mul1_op);
  sum0.set_attr_axes({1});
  sum0.set_attr_keep_dims(true);

  auto end_op1 = op::Square("end_op_1");
  end_op1.set_input_x(mul_op);
  auto end_op2 = op::Square("end_op_2");
  end_op2.set_input_x(sum0);

  std::vector<Operator> inputs{mul_input_x_data1,mul_input_x_data2, mul_input_x_data3};
  std::vector<Operator> outputs{end_op1, end_op2};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MulGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool ret = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConfusionMulGrad") {
      ret = true;
    }
  }
  EXPECT_EQ(ret, false);

}