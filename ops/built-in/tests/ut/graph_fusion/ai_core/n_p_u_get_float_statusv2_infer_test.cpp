#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "npu_loss_scale_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class n_p_u_get_float_status_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "n_p_u_get_float_status_v2 SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "n_p_u_get_float_status_v2 TearDown" << std::endl;
  }
};

TEST_F(n_p_u_get_float_status_v2_infer_test, n_p_u_get_float_status_v2_infer_test_1) {
  ge::Graph graph("n_p_u_get_float_status_v2_infer_test_1");

  // expect info
  std::vector<int64_t> expected_shape = {8};
  auto dtype = DT_FLOAT;

  // input
  ge::Shape output_shape({8});
  ge::TensorDesc desc_data(output_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_data.SetOriginFormat(ge::FORMAT_ND);
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);

  // new op
  auto test_op = op::NPUGetFloatStatusV2("NPUGetFloatStatusV2");
  test_op.create_dynamic_input_addr(1);
  test_op.set_dynamic_input_addr(0, data);
  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{test_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "NPUGetFloatStatusV2") {
      findOp = true;
      auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> dims = output_desc.GetShape().GetDims();
      EXPECT_EQ(dims, expected_shape);
      auto output_dtype = output_desc.GetDataType();
      EXPECT_EQ(output_dtype, dtype);
    }
  }
  EXPECT_EQ(findOp, true);

//  ge::OpDescPtr n_p_u_get_float_status_v2 = std::make_shared<ge::OpDesc>("NPUGetFloatStatusV2", "NPUGetFloatStatusV2");
//  n_p_u_get_float_status_v2->AddOutputDesc(desc_data);
//  n_p_u_get_float_status_v2->AddInputDesc(desc_data);
//  ge::NodePtr n_p_u_get_float_status_v2_node = graph->AddNode(n_p_u_get_float_status_v2);
//
//  ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("Const", "Const");
//  const1->AddOutputDesc(desc_data);
//  ge::NodePtr const_node = graph->AddNode(const1);
//  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), n_p_u_get_float_status_v2_node->GetInDataAnchor(0));
//
//  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");
//  netoutput->AddInputDesc(desc_data);
//
//  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
//  ge::GraphUtils::AddEdge(n_p_u_get_float_status_v2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
//  fe::FusionPassTestUtils::InferShapeAndType(graph);
}