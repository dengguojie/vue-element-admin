#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"


namespace fe {

class conv2d_to_fullyconnection_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_to_fullyconnection_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_to_fullyconnection_fusion_pass_test TearDown" << std::endl;
  }

  /*
   * ********** parent ********* sub **********
   *
   *             x1         x filter bias
   *              |          \  |  /
   *             func         conv2d
   *              |             |
   *           netoutput     netoutput
   *
   * *******************************************
   */
  void BuildGraphForSplit(ge::ComputeGraphPtr &parent_graph, ge::ComputeGraphPtr &sub_graph) {
    ge::GeShape bias_shape({64});
    ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    bias_desc.SetOriginFormat(ge::FORMAT_ND);
    bias_desc.SetOriginDataType(ge::DT_FLOAT);
    bias_desc.SetOriginShape(bias_shape);
    ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
    ge::OpDescPtr func = std::make_shared<ge::OpDesc>("func", "PartitionedCall");
    ge::OpDescPtr output = std::make_shared<ge::OpDesc>("output", "NetOutput");
    x1->AddInputDesc(bias_desc);
    x1->AddOutputDesc(bias_desc);
    func->AddOutputDesc(bias_desc);
    func->AddInputDesc(bias_desc);
    output->AddInputDesc(bias_desc);
    parent_graph = std::make_shared<ge::ComputeGraph>("parentgraph");
    ge::NodePtr x1_node = parent_graph->AddNode(x1);
    ge::NodePtr func_node = parent_graph->AddNode(func);
    ge::NodePtr output_node = parent_graph->AddNode(output);
    ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), func_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(func_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    ge::GeShape input_shape({1, 16, 32, 32});
    ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    x_desc.SetOriginFormat(ge::FORMAT_NCHW);
    x_desc.SetOriginDataType(ge::DT_FLOAT);
    x_desc.SetOriginShape(input_shape);
    ge::GeShape output_shape({1, 64, 1, 1});
    ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    y_desc.SetOriginFormat(ge::FORMAT_NCHW);
    y_desc.SetOriginDataType(ge::DT_FLOAT);
    y_desc.SetOriginShape(output_shape);
    ge::GeShape filter_shape({32, 32, 16, 64});
    ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
    filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
    filter_desc.SetOriginDataType(ge::DT_FLOAT);
    filter_desc.SetOriginShape(filter_shape);
    ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
    ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
    ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("bias", "Data");
    ge::OpDescPtr conv = std::make_shared<ge::OpDesc>("conv2d", "Conv2D");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");
    x->AddOutputDesc(x_desc);
    filter->AddOutputDesc(filter_desc);
    bias->AddInputDesc(bias_desc);
    bias->AddOutputDesc(bias_desc);
    conv->AddInputDesc("x", x_desc);
    conv->AddInputDesc("filter", filter_desc);
    conv->AddInputDesc("bias", bias_desc);
    conv->AddOutputDesc(y_desc);
    netoutput->AddInputDesc(y_desc);
    ge::AttrUtils::SetListInt(conv, "dilations", {1, 1, 1, 1});
    ge::AttrUtils::SetListInt(conv, "pads", {0, 0, 0, 0});
    ge::AttrUtils::SetListInt(conv, "strides", {1, 1, 1, 1});
    ge::AttrUtils::SetInt(conv, "groups", 1);
    sub_graph = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr x_node = sub_graph->AddNode(x);
    ge::NodePtr filter_node = sub_graph->AddNode(filter);
    ge::NodePtr bias_node = sub_graph->AddNode(bias);
    ge::NodePtr conv_node = sub_graph->AddNode(conv);
    ge::NodePtr netoutput_node = sub_graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(bias_node->GetOpDesc(), "_parent_node_index", 0);

    func_node->GetOpDesc()->AddSubgraphName("f");
    func_node->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
    sub_graph->SetParentNode(func_node);
    sub_graph->SetParentGraph(parent_graph);
  }

  /*
   * *******************************************
   *
   *            x filter bias
   *              \  |  /
   *               conv2d
   *                 |
   *              netoutput
   *
   * *******************************************
   */
  void BuildGraphForDynamicShape(ge::ComputeGraphPtr &sub_graph) {
    ge::GeShape bias_shape({64});
    ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    bias_desc.SetOriginFormat(ge::FORMAT_ND);
    bias_desc.SetOriginDataType(ge::DT_FLOAT);
    bias_desc.SetOriginShape(bias_shape);
    ge::GeShape input_shape({-1, 16, 32, 32});
    ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    x_desc.SetOriginFormat(ge::FORMAT_NCHW);
    x_desc.SetOriginDataType(ge::DT_FLOAT);
    x_desc.SetOriginShape(input_shape);
    ge::GeShape output_shape({-1, 64, 1, 1});
    ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    y_desc.SetOriginFormat(ge::FORMAT_NCHW);
    y_desc.SetOriginDataType(ge::DT_FLOAT);
    y_desc.SetOriginShape(output_shape);
    ge::GeShape filter_shape({32, 32, 16, 64});
    ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
    filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
    filter_desc.SetOriginDataType(ge::DT_FLOAT);
    filter_desc.SetOriginShape(filter_shape);
    ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
    ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
    ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("bias", "Data");
    ge::OpDescPtr conv = std::make_shared<ge::OpDesc>("conv2d", "Conv2D");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");
    x->AddOutputDesc(x_desc);
    filter->AddOutputDesc(filter_desc);
    bias->AddOutputDesc(bias_desc);
    conv->AddInputDesc("x", x_desc);
    conv->AddInputDesc("filter", filter_desc);
    conv->AddInputDesc("bias", bias_desc);
    conv->AddOutputDesc(y_desc);
    netoutput->AddInputDesc(y_desc);
    ge::AttrUtils::SetListInt(conv, "dilations", {1, 1, 1, 1});
    ge::AttrUtils::SetListInt(conv, "pads", {0, 0, 0, 0});
    ge::AttrUtils::SetListInt(conv, "strides", {1, 1, 1, 1});
    ge::AttrUtils::SetInt(conv, "groups", 1);
    sub_graph = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr x_node = sub_graph->AddNode(x);
    ge::NodePtr filter_node = sub_graph->AddNode(filter);
    ge::NodePtr bias_node = sub_graph->AddNode(bias);
    ge::NodePtr conv_node = sub_graph->AddNode(conv);
    ge::NodePtr netoutput_node = sub_graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  }

  /*
   * *******************************************
   *
   *            x filter bias
   *              \  |  /
   *               conv2d
   *                 |
   *              netoutput
   *
   * *******************************************
   */
  void BuildGraphForBiasNotDataType(ge::ComputeGraphPtr &sub_graph) {
    ge::GeShape bias_shape({64});
    ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    bias_desc.SetOriginFormat(ge::FORMAT_ND);
    bias_desc.SetOriginDataType(ge::DT_FLOAT);
    bias_desc.SetOriginShape(bias_shape);
    ge::GeShape input_shape({1, 16, 32, 32});
    ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    x_desc.SetOriginFormat(ge::FORMAT_NCHW);
    x_desc.SetOriginDataType(ge::DT_FLOAT);
    x_desc.SetOriginShape(input_shape);
    ge::GeShape output_shape({1, 64, 1, 1});
    ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    y_desc.SetOriginFormat(ge::FORMAT_NCHW);
    y_desc.SetOriginDataType(ge::DT_FLOAT);
    y_desc.SetOriginShape(output_shape);
    ge::GeShape filter_shape({32, 32, 16, 64});
    ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
    filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
    filter_desc.SetOriginDataType(ge::DT_FLOAT);
    filter_desc.SetOriginShape(filter_shape);
    ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
    ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
    ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("bias", "Const");
    ge::OpDescPtr conv = std::make_shared<ge::OpDesc>("conv2d", "Conv2D");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");
    x->AddOutputDesc(x_desc);
    filter->AddOutputDesc(filter_desc);
    bias->AddOutputDesc(bias_desc);
    conv->AddInputDesc("x", x_desc);
    conv->AddInputDesc("filter", filter_desc);
    conv->AddInputDesc("bias", bias_desc);
    conv->AddOutputDesc(y_desc);
    netoutput->AddInputDesc(y_desc);
    ge::AttrUtils::SetListInt(conv, "dilations", {1, 1, 1, 1});
    ge::AttrUtils::SetListInt(conv, "pads", {0, 0, 0, 0});
    ge::AttrUtils::SetListInt(conv, "strides", {1, 1, 1, 1});
    ge::AttrUtils::SetInt(conv, "groups", 1);
    sub_graph = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr x_node = sub_graph->AddNode(x);
    ge::NodePtr filter_node = sub_graph->AddNode(filter);
    ge::NodePtr bias_node = sub_graph->AddNode(bias);
    ge::NodePtr conv_node = sub_graph->AddNode(conv);
    ge::NodePtr netoutput_node = sub_graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  }
};

TEST_F(conv2d_to_fullyconnection_fusion_pass_test, input_bias_parent_graph_test) {
  ge::ComputeGraphPtr parent_graph;
  ge::ComputeGraphPtr sub_graph;
  BuildGraphForSplit(parent_graph, sub_graph);
  FusionPassTestUtils::RunGraphFusionPass("ConvToFullyConnectionFusionPass", fe::BUILT_IN_GRAPH_PASS, *sub_graph);
  bool find_fullyconnection_flag = false;
  for (auto node: sub_graph->GetAllNodes()) {
    if (node->GetType() == "FullyConnection") {
      find_fullyconnection_flag = true;
      break;
    }
  }
  EXPECT_EQ(find_fullyconnection_flag, true);
}

TEST_F(conv2d_to_fullyconnection_fusion_pass_test, dynamic_shape_not_supported_test) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraphForDynamicShape(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ConvToFullyConnectionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_fullyconnection_flag = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "FullyConnection") {
      find_fullyconnection_flag = true;
      break;
    }
  }
  EXPECT_EQ(find_fullyconnection_flag, false);
}

TEST_F(conv2d_to_fullyconnection_fusion_pass_test, bias_not_data_type_test) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraphForBiasNotDataType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ConvToFullyConnectionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_fullyconnection_flag = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "FullyConnection") {
      find_fullyconnection_flag = true;
      break;
    }
  }
  EXPECT_EQ(find_fullyconnection_flag, true);
}
} // namespace fe
