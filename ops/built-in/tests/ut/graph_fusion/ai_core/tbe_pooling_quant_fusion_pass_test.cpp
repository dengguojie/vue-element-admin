#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/lxfusion_json_util.h"
#include "common/lx_fusion_func.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/pooling/tbe_pooling_quant_fusion_pass.h"
#undef protected
#undef private
using namespace ge;

class TbePoolingQuantFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Tbe pool quant fusion pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Tbe pool quant fusion pass TearDown" << std::endl;
  }
};

TEST_F(TbePoolingQuantFusionTest, TbePoolingQuantFusionTest_1) {
  OpDescPtr pool_op = std::make_shared<OpDesc>("pool", "Pooling");
  OpDescPtr quant_op = std::make_shared<OpDesc>("quant", "AscendQuant");

  vector<int64_t> dim({1, 2, 255, 255, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NC1HWC0);
  tensor_desc.SetDataType(DT_FLOAT16);
  tensor_desc.SetShape(shape);

  pool_op->AddInputDesc("x", tensor_desc);
  pool_op->AddOutputDesc("y", tensor_desc);
  (void)ge::AttrUtils::SetStr(pool_op, "_op_slice_info", "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [0], \"tailOverLap\": [0]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}], \"reduceMaps\": [], \"l1FusionEnable\": 2, \"minTbeL1Space\": 0}}");
  quant_op->AddInputDesc("x", tensor_desc);
  quant_op->AddOutputDesc("y", tensor_desc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr pool_node = graph->AddNode(pool_op);
  NodePtr quant_node = graph->AddNode(quant_op);
  ge::GraphUtils::AddEdge(pool_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));

  fe::Pool2dQuantFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc pool_desc;
  pool_desc.desc_name = "Pool2d";
  const fe::BufferFusionOpDesc *pool_desc_ptr = &pool_desc;
  fe::BufferFusionOpDesc quant_desc;
  quant_desc.desc_name = "quant";
  const fe::BufferFusionOpDesc *quant_desc_ptr = &quant_desc;

  fe::BufferFusionMapping mapping;
  vector<NodePtr> pool_nodes = {pool_node};
  mapping.emplace(pool_desc_ptr, pool_nodes);
  vector<NodePtr> quant_nodes = {quant_node};
  mapping.emplace(quant_desc_ptr, quant_nodes);

  vector<NodePtr> fusion_nodes = {pool_node, quant_node};

  for (const auto &item : mapping) {
    const fe::BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr) {
      std::cout << "op_desc->desc_name:" << op_desc->desc_name << std::endl;
    }
  }

  fusion_pass.SetSplitInfo(mapping, fusion_nodes);

  for (auto fusion_node : fusion_nodes) {
    std::string op_calc_info_str;
    (void)ge::AttrUtils::GetStr(fusion_node->GetOpDesc(), "_fusion_op_slice_info", op_calc_info_str);
    std::cout << "op_calc_info_str:" << op_calc_info_str << std::endl;
  }

}
