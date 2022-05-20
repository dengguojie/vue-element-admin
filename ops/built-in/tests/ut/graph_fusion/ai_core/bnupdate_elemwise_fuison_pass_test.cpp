#include "gtest/gtest.h"
#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "common/lx_fusion_func.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/bnupdate/bnupdate_eltwise_eltwise_fusion_pass.h"
#include "buffer_fusion/ub_fusion/ai_core/bnupdate/bnupdate_eltwise_fusion_pass.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "common/util/platform_info.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class bnupdate_eltwise_ub_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "bnupdate_eltwise_ub_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "bnupdate_eltwise_ub_fusion_test TearDown" << std::endl;
    }
};

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
    ge::GeTensorDesc desc_##name(shape_out, format_out, dtype); \
    desc_##name.SetOriginFormat(format_in);                \
    desc_##name.SetOriginShape(shape_in)

namespace fe {
  static Status RunBnEltWiseFusionPass(string fusion_pass_name, ge::ComputeGraphPtr& compute_graph_ptr) {
    std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
        BufferFusionPassRegistry::GetInstance().GetCreateFnByType(fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
    const auto &iter = create_fns.find(fusion_pass_name);
    auto buffer_fusion_pass_base_ptr = std::unique_ptr<BnupdateEltwiseFusionPass>(
                    dynamic_cast<BnupdateEltwiseFusionPass *>(iter->second()));
    if (buffer_fusion_pass_base_ptr == nullptr) {
        return FAILED;
    }
    ge::ComputeGraph::Vistor<ge::NodePtr> node_ptrs = compute_graph_ptr->GetAllNodes();
    buffer_fusion_pass_base_ptr->SetName(fusion_pass_name);
    vector<BufferFusionPattern*> patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
    for (auto pattern : patterns) {
      std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
      vector<ge::NodePtr> BnNodes;
      for (auto i : node_ptrs) {
        auto opDesc = i->GetOpDesc();
        if (opDesc->GetType() == "BNTrainingUpdate") {
          BnNodes.push_back(i);
        }
      }
      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "bnupdate") {
          mapping[i] = BnNodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      buffer_fusion_pass_base_ptr->GetFusionNodes(mapping, fusion_nodes);
      EXPECT_EQ(fusion_nodes.size(), 1);
    }
    return SUCCESS;
  }

  static Status RunBnEltEltWiseFusionPass(string fusion_pass_name, ge::ComputeGraphPtr& compute_graph_ptr) {
    std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
        BufferFusionPassRegistry::GetInstance().GetCreateFnByType(fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
    const auto &iter = create_fns.find(fusion_pass_name);
    auto buffer_fusion_pass_base_ptr = std::unique_ptr<BnupdateEltwiseEltwiseFusionPass>(
                    dynamic_cast<BnupdateEltwiseEltwiseFusionPass *>(iter->second()));
    if (buffer_fusion_pass_base_ptr == nullptr) {
        return FAILED;
    }
    ge::ComputeGraph::Vistor<ge::NodePtr> node_ptrs = compute_graph_ptr->GetAllNodes();
    buffer_fusion_pass_base_ptr->SetName(fusion_pass_name);
    vector<BufferFusionPattern*> patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
    for (auto pattern : patterns) {
      std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
      vector<ge::NodePtr> BnNodes;
      for (auto i : node_ptrs) {
        auto opDesc = i->GetOpDesc();
        if (opDesc->GetType() == "BNTrainingUpdate") {
          BnNodes.push_back(i);
        }
      }
      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "bnupdate") {
          mapping[i] = BnNodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      buffer_fusion_pass_base_ptr->GetFusionNodes(mapping, fusion_nodes);
      EXPECT_EQ(fusion_nodes.size(), 1);
    }
    return SUCCESS;
  }

}

TEST_F(bnupdate_eltwise_ub_fusion_test, bnupdate_eltwise_ub_fusion_test_1) {
  ge::Graph graph("bnupdate_eltwise_ub_fusion_test_1");
  DESC_DATA(data_a, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);
  ge::OpDescPtr bnupdate = std::make_shared<ge::OpDesc>("bnupdate", "BNTrainingUpdate");
  ge::OpDescPtr e1 = std::make_shared<ge::OpDesc>("e1", "Elemwise");
  ge::OpDescPtr o1 = std::make_shared<ge::OpDesc>("o1", "Output");
  ge::OpDescPtr o2 = std::make_shared<ge::OpDesc>("o2", "Output");
  ge::OpDescPtr o3 = std::make_shared<ge::OpDesc>("o3", "Output");
  ge::OpDescPtr o4 = std::make_shared<ge::OpDesc>("o4", "Output");
  ge::OpDescPtr o5 = std::make_shared<ge::OpDesc>("o5", "Output");
  ge::OpDescPtr o6 = std::make_shared<ge::OpDesc>("o6", "Output");
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  e1->AddInputDesc(desc_data_a);
  o1->AddInputDesc(desc_data_a);
  o2->AddInputDesc(desc_data_a);
  o3->AddInputDesc(desc_data_a);
  o4->AddInputDesc(desc_data_a);
  o5->AddInputDesc(desc_data_a);
  o6->AddInputDesc(desc_data_a);
  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
  ge::NodePtr bnupdate_node = compute_graph_ptr->AddNode(bnupdate);
  ge::NodePtr e1_node = compute_graph_ptr->AddNode(e1);
  ge::NodePtr o1_node = compute_graph_ptr->AddNode(o1);
  ge::NodePtr o2_node = compute_graph_ptr->AddNode(o2);
  ge::NodePtr o3_node = compute_graph_ptr->AddNode(o3);
  ge::NodePtr o4_node = compute_graph_ptr->AddNode(o4);
  ge::NodePtr o5_node = compute_graph_ptr->AddNode(o5);
  ge::NodePtr o6_node = compute_graph_ptr->AddNode(o6);
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(0), e1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(1), o1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(2), o2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(3), o3_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(4), o4_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(5), o5_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(6), o6_node->GetInDataAnchor(0));
  Status res = fe::RunBnEltWiseFusionPass("BNUpdateEltwiseFusionPass", compute_graph_ptr);
}

TEST_F(bnupdate_eltwise_ub_fusion_test, bnupdate_eltwise_ub_fusion_test_2) {
  ge::Graph graph("bnupdate_eltwise_ub_fusion_test_2");
  DESC_DATA(data_a, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);
  ge::OpDescPtr bnupdate = std::make_shared<ge::OpDesc>("bnupdate", "BNTrainingUpdate");
  ge::OpDescPtr e1 = std::make_shared<ge::OpDesc>("e1", "Elemwise");
  ge::OpDescPtr e2 = std::make_shared<ge::OpDesc>("e2", "Elemwise");
  ge::OpDescPtr o1 = std::make_shared<ge::OpDesc>("o1", "Output");
  ge::OpDescPtr o2 = std::make_shared<ge::OpDesc>("o2", "Output");
  ge::OpDescPtr o3 = std::make_shared<ge::OpDesc>("o3", "Output");
  ge::OpDescPtr o4 = std::make_shared<ge::OpDesc>("o4", "Output");
  ge::OpDescPtr o5 = std::make_shared<ge::OpDesc>("o5", "Output");
  ge::OpDescPtr o6 = std::make_shared<ge::OpDesc>("o6", "Output");
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  bnupdate->AddOutputDesc(desc_data_a);
  e1->AddInputDesc(desc_data_a);
  e1->AddOutputDesc(desc_data_a);
  e2->AddInputDesc(desc_data_a);
  o1->AddInputDesc(desc_data_a);
  o2->AddInputDesc(desc_data_a);
  o3->AddInputDesc(desc_data_a);
  o4->AddInputDesc(desc_data_a);
  o5->AddInputDesc(desc_data_a);
  o6->AddInputDesc(desc_data_a);
  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
  ge::NodePtr bnupdate_node = compute_graph_ptr->AddNode(bnupdate);
  ge::NodePtr e1_node = compute_graph_ptr->AddNode(e1);
  ge::NodePtr e2_node = compute_graph_ptr->AddNode(e2);
  ge::NodePtr o1_node = compute_graph_ptr->AddNode(o1);
  ge::NodePtr o2_node = compute_graph_ptr->AddNode(o2);
  ge::NodePtr o3_node = compute_graph_ptr->AddNode(o3);
  ge::NodePtr o4_node = compute_graph_ptr->AddNode(o4);
  ge::NodePtr o5_node = compute_graph_ptr->AddNode(o5);
  ge::NodePtr o6_node = compute_graph_ptr->AddNode(o6);
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(0), e1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(1), o1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(2), o2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(3), o3_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(4), o4_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(5), o5_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(bnupdate_node->GetOutDataAnchor(6), o6_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(e1_node->GetOutDataAnchor(0), e2_node->GetInDataAnchor(0));
  Status res = fe::RunBnEltEltWiseFusionPass("BNUpdateEltwiseEltwiseFusionPass", compute_graph_ptr);
}

