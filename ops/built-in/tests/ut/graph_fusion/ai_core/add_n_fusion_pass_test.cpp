#include "nn_batch_norm_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class add_n_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "inplace_add SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(add_n_fusion_test, add_n_fusion_test_1) {
  auto graph = std::make_shared<ge::ComputeGraph>("add_n_fusion_test_1");

  ge::GeShape bias_shape({64});
  ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  bias_desc.SetOriginFormat(ge::FORMAT_NHWC);
  bias_desc.SetOriginDataType(ge::DT_FLOAT);
  bias_desc.SetOriginShape(bias_shape);

  ge::OpDescPtr addn = std::make_shared<ge::OpDesc>("AddN", "AddN");
  addn->AddOutputDesc(bias_desc);
  for (int i = 0; i < 65; ++i) {
    addn->AddInputDesc(bias_desc);
  }
  ge::NodePtr addn_node = graph->AddNode(addn);

  for (int i = 0; i < 65; ++i) {
    ge::OpDescPtr const1 = std::make_shared<ge::OpDesc>("Const" + i, "Const");
    const1->AddOutputDesc(bias_desc);
    ge::NodePtr const_node = graph->AddNode(const1);
    ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), addn_node->GetInDataAnchor(i));
  }
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");
  netoutput->AddInputDesc(bias_desc);

  ge::NodePtr netoutput_node = graph->AddNode(netoutput);
  ge::GraphUtils::AddEdge(addn_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  fe::FusionPassTestUtils::InferShapeAndType(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("AddNFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
}