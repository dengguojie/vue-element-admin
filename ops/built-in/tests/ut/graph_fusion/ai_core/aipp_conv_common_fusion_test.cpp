#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_common_fusion_pass.h"
#undef protected
#undef private

using namespace ge;

class AippConvCommonFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Aipp conv common fusion pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Aipp conv common fusion pass TearDown" << std::endl;
  }
};

TEST_F(AippConvCommonFusionTest, AippConvCommonFusionTest_1) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr sqrt_op = std::make_shared<OpDesc>("sqrt", "Sqrt");
  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");

  vector<int64_t> dim({40, 25, 7, 7});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShape(shape);

  relu_op->AddInputDesc("x", tensor_desc);
  relu_op->AddOutputDesc("y", tensor_desc);
  sqrt_op->AddInputDesc("x", tensor_desc);
  sqrt_op->AddOutputDesc("y", tensor_desc);
  relu6_op->AddInputDesc("x", tensor_desc);
  relu6_op->AddOutputDesc("y", tensor_desc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sqrt_node = graph->AddNode(sqrt_op);
  NodePtr relu6_node = graph->AddNode(relu6_op);
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(sqrt_node->GetOutDataAnchor(0), relu6_node->GetInDataAnchor(0));

  fe::TbeAippCommonFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc fusion_desc;
  fusion_desc.desc_name = "elemwise";
  const fe::BufferFusionOpDesc *fusion_desc_ptr = &fusion_desc;
  fe::BufferFusionMapping mapping;
  vector<NodePtr> elemwise_nodes = {relu_node, sqrt_node, relu6_node};
  mapping.emplace(fusion_desc_ptr, elemwise_nodes);
  vector<NodePtr> fusion_nodes;

  fe::Status status = fusion_pass.GetFusionNodes(mapping, fusion_nodes);
  EXPECT_EQ(fusion_nodes.size(), 1);
  fusion_desc_ptr = nullptr;
}
