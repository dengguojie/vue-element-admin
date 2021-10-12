#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "state_ops.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "graph_fusion/ai_core/conv2d_group_fusion_pass.h"

using namespace ge;
using namespace op;
using namespace fe;

class conv2d_group_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_group_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_group_fusion_test TearDown" << std::endl;
  }
};

/* (const) + conv2d + add */
TEST_F(conv2d_group_fusion_test, conv2d_group_fusion_test_1) {
  ge::Graph graph("conv2d_group_fusion_test_1");

  auto conv_input_x_data = op::Data("conv_input_x_data");
  std::vector<int64_t> dims_x{1, 128, 56, 56};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_INT8);
  tensorDescX.SetOriginShape(shape_x);
  tensorDescX.SetOriginFormat(FORMAT_NCHW);
  conv_input_x_data.update_input_desc_x(tensorDescX);
  conv_input_x_data.update_output_desc_y(tensorDescX);

  auto conv_input_filter_data = op::Const("conv_input_filter_data");
  auto conv_input_bias_data = op::Const("conv_input_bias_data");

  Tensor conv_input_filter_tensor;
  float *conv_input_filter_tensor_value = new float[128 * 4 * 3 * 3];
  for (int i = 0; i < 128 * 4 * 3 * 3; i++) {
    *(conv_input_filter_tensor_value + i) = 0.1;
  }
  conv_input_filter_tensor.SetData((uint8_t *) conv_input_filter_tensor_value, 128 * 4 * 3 * 3 * 4);

  std::vector<int64_t> dims_filter{128, 4, 3, 3};
  ge::Shape shape_filter(dims_filter);
  ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_INT8);
  tensorDescFilter.SetOriginShape(shape_filter);
  tensorDescFilter.SetOriginFormat(FORMAT_NCHW);
  conv_input_filter_tensor.SetTensorDesc(tensorDescFilter);
  conv_input_filter_data.set_attr_value(conv_input_filter_tensor);

  Tensor dims_bias_tensor;
  float *dims_bias_tensor_value = new float[128];
  for (int i = 0; i < 128; i++) {
    *(dims_bias_tensor_value + i) = 0.1;
  }
  dims_bias_tensor.SetData((uint8_t *) dims_bias_tensor_value, 128 * 4);

  std::vector<int64_t> dims_bias{128};
  ge::Shape shape_bias(dims_bias);
  ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_INT32);
  tensorDescBias.SetOriginShape(shape_bias);
  tensorDescBias.SetOriginFormat(FORMAT_NCHW);
  dims_bias_tensor.SetTensorDesc(tensorDescBias);
  conv_input_bias_data.set_attr_value(dims_bias_tensor);

  auto conv_op = op::Conv2D("conv_1");
  conv_op.set_input_x(conv_input_x_data)
  .set_input_filter(conv_input_filter_data)
  .set_input_bias(conv_input_bias_data);
  conv_op.set_attr_data_format("NCHW");
  conv_op.set_attr_strides({1, 1, 1, 1});
  conv_op.set_attr_pads({1, 1, 1, 1});
  conv_op.set_attr_groups(32);

  auto add_shape = ge::Shape({1,128,56,56});
  TensorDesc desc_add(add_shape, FORMAT_NCHW, DT_INT32);
  Tensor add_tensor(desc_add);

  auto add_op = op::Const("add_op")
          .set_attr_value(add_tensor);

  auto end_op1 = op::Add("Conv2D/add");
  end_op1.set_input_x1(conv_op)
  .set_input_x2(add_op);

  std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, add_op};
  std::vector<Operator> outputs{end_op1};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("GroupConv2DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool splitFlag = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "SplitD") {
      splitFlag = true;
    }
  }
  EXPECT_EQ(splitFlag, false);

  delete[] dims_bias_tensor_value;
  delete[] conv_input_filter_tensor_value;
}

TEST_F(conv2d_group_fusion_test, conv2d_group_fusion_test_SwapNumChn_and_GetGroups) {
  vector<int64_t> dim = {4, 3, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  relu_op->AddInputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  Conv2DGroupFusionPass pass;
  fe::Status status = pass.SwapNumChn(relu_op, true, 0);
  EXPECT_EQ(status, fe::SUCCESS);
  status = pass.SwapNumChn(relu_op, false, 0);
  EXPECT_EQ(status, fe::SUCCESS);
  int64_t group = pass.GetGroups(relu_op);
  EXPECT_EQ(group, 1);
  ge::AttrUtils::SetInt(relu_op, "groups", 8);
  group = pass.GetGroups(relu_op);
  EXPECT_EQ(group, 8);
}

TEST_F(conv2d_group_fusion_test, conv2d_group_fusion_test_ProcessDepthwiseConv_1) {
  vector<int64_t> dim = {4, 1, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddOutputDesc(tenosr_desc);
  OpDescPtr filter_op = std::make_shared<OpDesc>("filter", "Const");
  filter_op->AddInputDesc(tenosr_desc);
  filter_op->AddOutputDesc(tenosr_desc);

  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  ge::NodePtr conv_node = graph_out->AddNode(conv_op);
  ge::NodePtr filter_node = graph_out->AddNode(filter_op);
  GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  Conv2DGroupFusionPass pass;
  fe::Status status = pass.ProcessDepthwiseConv(conv_node);
}

TEST_F(conv2d_group_fusion_test, conv2d_group_fusion_test_ProcessDepthwiseConv_2) {
  vector<int64_t> dim = {4, 3, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddOutputDesc(tenosr_desc);
  OpDescPtr filter_op = std::make_shared<OpDesc>("filter", "Const");
  filter_op->AddInputDesc(tenosr_desc);
  filter_op->AddOutputDesc(tenosr_desc);
  OpDescPtr fm_op = std::make_shared<OpDesc>("fm", "Data");
  fm_op->AddInputDesc(tenosr_desc);
  fm_op->AddOutputDesc(tenosr_desc);

  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  ge::NodePtr conv_node = graph_out->AddNode(conv_op);
  ge::NodePtr filter_node = graph_out->AddNode(filter_op);
  ge::NodePtr fm_node = graph_out->AddNode(fm_op);
  GraphUtils::AddEdge(fm_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  Conv2DGroupFusionPass pass;
  fe::Status status = pass.ProcessDepthwiseConv(conv_node);
}
