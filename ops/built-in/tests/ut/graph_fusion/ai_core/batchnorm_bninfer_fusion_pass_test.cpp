#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class batchnorm_bninfer_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "inplace_add SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_1) {
  ge::Graph graph("batchnorm_bninfer_fusion_test_1");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto input1 = op::Const("intput1");
  Tensor axis;
  float *dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  axis.SetData((uint8_t *)dataValue, 4);
  input1.set_attr_value(axis);
  input1.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  delete [] dataValue;

  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input3 = op::Const("input3");
  input3.set_attr_value(axis);
  input3.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input4 = op::Const("input4");
  input4.set_attr_value(axis);
  input4.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto batchnorm = op::BatchNorm("batchnorm");
  batchnorm.set_input_x(input0);
  batchnorm.set_input_scale(input1);
  batchnorm.set_input_offset(input2);
  batchnorm.set_input_mean(input3);
  batchnorm.set_input_variance(input4);
  batchnorm.set_attr_is_training(false);

  std::vector<Operator> inputs{input0, input1, input2, input3, input4};
  std::vector<Operator> outputs{batchnorm};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_1_after");
}

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_2) {
  ge::Graph graph("batchnorm_bninfer_fusion_test_2");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto input1 = op::Const("input1");
  Tensor axis;
  float *dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  axis.SetData((uint8_t *)dataValue, 4);
  input1.set_attr_value(axis);
  input1.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  delete[] dataValue;

  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input3 = op::Const("input3");
  input3.set_attr_value(axis);
  input3.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input4 = op::Data("input4");
  input4.update_input_desc_x(tensorDescMs);
  input4.update_output_desc_y(tensorDescMs);

  auto batchnorm = op::BatchNorm("batchnorm");
  batchnorm.set_input_x(input0);
  batchnorm.set_input_scale(input1);
  batchnorm.set_input_offset(input2);
  batchnorm.set_input_mean(input3);
  batchnorm.set_input_variance(input4);
  batchnorm.set_attr_is_training(false);

  std::vector<Operator> inputs{input0, input1, input2, input3, input4};
  std::vector<Operator> outputs{batchnorm};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_2_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_2_after");
}

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_3) {
  auto graph = std::make_shared<ge::ComputeGraph>("batchnorm_bninfer_fusion_test_3");

  ge::GeShape bias_shape({64});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
  bias_desc.SetOriginDataType(ge::DT_FLOAT);
  bias_desc.SetOriginShape(bias_shape);

  ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
  ge::OpDescPtr scale = std::make_shared<ge::OpDesc>("scale", "Data");
  ge::OpDescPtr offset = std::make_shared<ge::OpDesc>("offset", "Const");
  ge::OpDescPtr mean = std::make_shared<ge::OpDesc>("mean", "Const");
  ge::OpDescPtr variance = std::make_shared<ge::OpDesc>("variance", "Const");
  ge::OpDescPtr batchnorm = std::make_shared<ge::OpDesc>("batchnorm", "BatchNorm");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  x1->AddOutputDesc(bias_desc);
  scale->AddOutputDesc(bias_desc);
  offset->AddOutputDesc(bias_desc);
  mean->AddOutputDesc(bias_desc);
  variance->AddOutputDesc(bias_desc);

  batchnorm->AddInputDesc(bias_desc);
  batchnorm->AddInputDesc(bias_desc);
  batchnorm->AddInputDesc(bias_desc);
  batchnorm->AddInputDesc(bias_desc);
  batchnorm->AddInputDesc(bias_desc);
  batchnorm->AddOutputDesc(bias_desc);
  batchnorm->AddOutputDesc(bias_desc);
  batchnorm->AddOutputDesc(bias_desc);
  batchnorm->AddOutputDesc(bias_desc);
  batchnorm->AddOutputDesc(bias_desc);

  netoutput->AddInputDesc(bias_desc);

  ge::NodePtr x1_node = graph->AddNode(x1);
  ge::NodePtr scale_node = graph->AddNode(scale);
  ge::NodePtr offset_node = graph->AddNode(offset);
  ge::NodePtr mean_node = graph->AddNode(mean);
  ge::NodePtr variance_node = graph->AddNode(variance);
  ge::NodePtr batchnorm_node = graph->AddNode(batchnorm);
  ge::NodePtr netoutput_node = graph->AddNode(netoutput);

  ge::AttrUtils::SetBool(batchnorm_node->GetOpDesc(), "is_training", false);
  ge::AttrUtils::SetFloat(batchnorm_node->GetOpDesc(), "epsilon", 1.1);

  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(scale_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(offset_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(variance_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(batchnorm_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(batchnorm_node->GetOutControlAnchor(), netoutput_node->GetInControlAnchor());

  std::cout << batchnorm_node->GetAllOutDataAnchorsSize() << std::endl;

  EXPECT_EQ(batchnorm_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);

  std::cout << batchnorm_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() << std::endl;
  fe::FusionPassTestUtils::InferShapeAndType(graph);
  GE_DUMP(graph, "batchnorm_bninfer_fusion_test_2_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
  GE_DUMP(graph, "batchnorm_bninfer_fusion_test_2_after");
}

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_4) {

auto graph = std::make_shared<ge::ComputeGraph>("batchnorm_bninfer_fusion_test_4");

ge::GeShape bias_shape({64});
ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
bias_desc.SetOriginDataType(ge::DT_FLOAT);
bias_desc.SetOriginShape(bias_shape);

ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
ge::OpDescPtr scale = std::make_shared<ge::OpDesc>("scale", "Const");
ge::OpDescPtr offset = std::make_shared<ge::OpDesc>("offset", "Const");
ge::OpDescPtr mean = std::make_shared<ge::OpDesc>("mean", "Const");
ge::OpDescPtr variance = std::make_shared<ge::OpDesc>("variance", "Const");
ge::OpDescPtr batchnorm = std::make_shared<ge::OpDesc>("batchnorm", "BatchNorm");
ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

x1->AddOutputDesc(bias_desc);
scale->AddOutputDesc(bias_desc);
offset->AddOutputDesc(bias_desc);
mean->AddOutputDesc(bias_desc);
variance->AddOutputDesc(bias_desc);

batchnorm->AddInputDesc(bias_desc);
batchnorm->AddInputDesc(bias_desc);
batchnorm->AddInputDesc(bias_desc);
batchnorm->AddInputDesc(bias_desc);
batchnorm->AddInputDesc(bias_desc);
batchnorm->AddOutputDesc(bias_desc);
batchnorm->AddOutputDesc(bias_desc);
batchnorm->AddOutputDesc(bias_desc);
batchnorm->AddOutputDesc(bias_desc);
batchnorm->AddOutputDesc(bias_desc);


netoutput->AddInputDesc(bias_desc);

ge::NodePtr x1_node = graph->AddNode(x1);
ge::NodePtr scale_node = graph->AddNode(scale);
ge::NodePtr offset_node = graph->AddNode(offset);
ge::NodePtr mean_node = graph->AddNode(mean);
ge::NodePtr variance_node = graph->AddNode(variance);
ge::NodePtr batchnorm_node = graph->AddNode(batchnorm);
ge::NodePtr netoutput_node = graph->AddNode(netoutput);

ge::AttrUtils::SetBool(batchnorm_node->GetOpDesc(), "is_training", false);
ge::AttrUtils::SetFloat(batchnorm_node->GetOpDesc(), "epsilon", 1.1);

ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(0));
ge::GraphUtils::AddEdge(scale_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(1));
ge::GraphUtils::AddEdge(offset_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(2));
ge::GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(3));
ge::GraphUtils::AddEdge(variance_node->GetOutDataAnchor(0), batchnorm_node->GetInDataAnchor(4));
ge::GraphUtils::AddEdge(batchnorm_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
ge::GraphUtils::AddEdge(batchnorm_node->GetOutControlAnchor(), netoutput_node->GetInControlAnchor());

std::cout << batchnorm_node->GetAllOutDataAnchorsSize()<< std::endl;

EXPECT_EQ(batchnorm_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);

std::cout << batchnorm_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() << std::endl;
  fe::FusionPassTestUtils::InferShapeAndType(graph);
  GE_DUMP(graph, "batchnorm_bninfer_fusion_test_2_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *graph);
  GE_DUMP(graph, "batchnorm_bninfer_fusion_test_2_after");
}