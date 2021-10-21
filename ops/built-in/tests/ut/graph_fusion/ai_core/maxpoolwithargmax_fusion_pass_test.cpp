#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class maxpoolwithargmax_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "matrix_diag_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matrix_diag_fusion_test TearDown" << std::endl;
  }
};
TEST_F(maxpoolwithargmax_fusion_pass_test, maxpoolwithargmax_fusion_pass_test_1) {
  auto graph = std::make_shared<ge::ComputeGraph>("maxpoolwithargmax_fusion_pass_test_1");

  ge::GeShape bias_shape({64,32,32,32});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_INT64);
  bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
  bias_desc.SetOriginDataType(ge::DT_INT64);
  bias_desc.SetOriginShape(bias_shape);

  ge::OpDescPtr input1 = std::make_shared<ge::OpDesc>("input1", "Data");
  ge::OpDescPtr input2 = std::make_shared<ge::OpDesc>("input2", "Data");
  ge::OpDescPtr input3 = std::make_shared<ge::OpDesc>("input3", "Data");
  ge::OpDescPtr output1 = std::make_shared<ge::OpDesc>("output1", "Data");
  ge::OpDescPtr output2 = std::make_shared<ge::OpDesc>("output2", "MaxPoolGradWithArgmax");
  ge::OpDescPtr maxpoolwithargmax0 = std::make_shared<ge::OpDesc>("maxpoolwithargmax0", "MaxPoolWithArgmax");

  input1->AddOutputDesc(bias_desc);
  input2->AddOutputDesc(bias_desc);
  input3->AddOutputDesc(bias_desc);

  maxpoolwithargmax0->AddInputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);

  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "ksize", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "strides", {1, 1, 1, 1});
  ge::AttrUtils::SetStr(maxpoolwithargmax0, "padding", "VALID");

  output1->AddInputDesc(bias_desc);

  output2->AddInputDesc(bias_desc);
  output2->AddInputDesc(bias_desc);
  output2->AddInputDesc(bias_desc);
  output2->AddOutputDesc(bias_desc);
  ge::AttrUtils::SetListInt(output2, "ksize", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(output2, "strides", {1, 1, 1, 1});
  ge::AttrUtils::SetStr(output2, "padding", "VALID");

  ge::NodePtr input1_node = graph->AddNode(input1);
  ge::NodePtr input2_node = graph->AddNode(input2);
  ge::NodePtr input3_node = graph->AddNode(input3);
  ge::NodePtr output1_node = graph->AddNode(output1);
  ge::NodePtr output2_node = graph->AddNode(output2);
  ge::NodePtr maxpoolwithargmax0_node = graph->AddNode(maxpoolwithargmax0);

  ge::GraphUtils::AddEdge(input1_node->GetOutDataAnchor(0), maxpoolwithargmax0_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(0), output1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(1), output2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(input2_node->GetOutDataAnchor(0), output2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(input3_node->GetOutDataAnchor(0), output2_node->GetInDataAnchor(2));

  fe::FusionPassTestUtils::InferShapeAndType(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("MaxPoolWithArgmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool avgPoolMatch = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, false);
}

TEST_F(maxpoolwithargmax_fusion_pass_test, maxpoolwithargmax_fusion_pass_test_2) {
  auto graph = std::make_shared<ge::ComputeGraph>("maxpoolwithargmax_fusion_pass_test_2");

  ge::GeShape bias_shape({64,32,32,32});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_INT64);
  bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
  bias_desc.SetOriginDataType(ge::DT_INT64);
  bias_desc.SetOriginShape(bias_shape);

  ge::OpDescPtr input1 = std::make_shared<ge::OpDesc>("input1", "Data");
  ge::OpDescPtr output1 = std::make_shared<ge::OpDesc>("output1", "Data");
  ge::OpDescPtr output2 = std::make_shared<ge::OpDesc>("output2", "Data");
  ge::OpDescPtr maxpoolwithargmax0 = std::make_shared<ge::OpDesc>("maxpoolwithargmax0", "MaxPoolWithArgmax");

  input1->AddOutputDesc(bias_desc);

  maxpoolwithargmax0->AddInputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);

  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "ksize", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "strides", {1, 1, 1, 1});
  ge::AttrUtils::SetStr(maxpoolwithargmax0, "padding", "SAME");

  output1->AddInputDesc(bias_desc);
  output2->AddInputDesc(bias_desc);

  ge::NodePtr input1_node = graph->AddNode(input1);
  ge::NodePtr output1_node = graph->AddNode(output1);
  ge::NodePtr output2_node = graph->AddNode(output2);
  ge::NodePtr maxpoolwithargmax0_node = graph->AddNode(maxpoolwithargmax0);

  ge::GraphUtils::AddEdge(input1_node->GetOutDataAnchor(0), maxpoolwithargmax0_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(0), output1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(1), output2_node->GetInDataAnchor(0));

  fe::FusionPassTestUtils::InferShapeAndType(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("MaxPoolWithArgmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool avgPoolMatch = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(maxpoolwithargmax_fusion_pass_test, maxpoolwithargmax_fusion_pass_test_3) {
  auto graph = std::make_shared<ge::ComputeGraph>("maxpoolwithargmax_fusion_pass_test_3");

  ge::GeShape bias_shape({64,32,32,32});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NCHW, ge::DT_INT64);
  bias_desc.SetOriginFormat(ge::FORMAT_NCHW);
  bias_desc.SetOriginDataType(ge::DT_INT64);
  bias_desc.SetOriginShape(bias_shape);

  ge::OpDescPtr input1 = std::make_shared<ge::OpDesc>("input1", "Data");
  ge::OpDescPtr output1 = std::make_shared<ge::OpDesc>("output1", "Data");
  ge::OpDescPtr output2 = std::make_shared<ge::OpDesc>("output2", "Data");
  ge::OpDescPtr maxpoolwithargmax0 = std::make_shared<ge::OpDesc>("maxpoolwithargmax0", "MaxPoolWithArgmax");

  input1->AddOutputDesc(bias_desc);

  maxpoolwithargmax0->AddInputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);
  maxpoolwithargmax0->AddOutputDesc(bias_desc);

  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "ksize", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(maxpoolwithargmax0, "strides", {1, 1, 1, 1});
  ge::AttrUtils::SetStr(maxpoolwithargmax0, "padding", "SAME");

  output1->AddInputDesc(bias_desc);
  output2->AddInputDesc(bias_desc);

  ge::NodePtr input1_node = graph->AddNode(input1);
  ge::NodePtr output1_node = graph->AddNode(output1);
  ge::NodePtr output2_node = graph->AddNode(output2);
  ge::NodePtr maxpoolwithargmax0_node = graph->AddNode(maxpoolwithargmax0);

  ge::GraphUtils::AddEdge(input1_node->GetOutDataAnchor(0), maxpoolwithargmax0_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(0), output1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(maxpoolwithargmax0_node->GetOutDataAnchor(1), output2_node->GetInDataAnchor(0));

  fe::FusionPassTestUtils::InferShapeAndType(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("MaxPoolWithArgmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool avgPoolMatch = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}