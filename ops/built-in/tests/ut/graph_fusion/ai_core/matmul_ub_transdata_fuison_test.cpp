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
#include "buffer_fusion/ub_fusion/ai_core/matmul/matmul_transdata_ub_fusion.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class matmul_transdata_ub_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "matmul_transdata_ub_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "matmul_transdata_ub_fusion_test TearDown" << std::endl;
    }
};

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
    ge::GeTensorDesc desc_##name(shape_out, format_out, dtype); \
    desc_##name.SetOriginFormat(format_in);                \
    desc_##name.SetOriginShape(shape_in)

namespace fe {
  static Status RunBufferFusionPass(string fusion_pass_name, BufferFusionPassType pass_type,
                                    ge::ComputeGraphPtr& compute_graph_ptr) {
      std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
          BufferFusionPassRegistry::GetInstance().GetCreateFnByType(pass_type);
      const auto &iter = create_fns.find(fusion_pass_name);
      if (iter != create_fns.end()) {
          if (pass_type == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto buffer_fusion_pass_base_ptr = std::unique_ptr<MatmulTransdataFusionPass>(
                      dynamic_cast<MatmulTransdataFusionPass *>(iter->second()));
              if (buffer_fusion_pass_base_ptr == nullptr) {
                  return FAILED;
              }
              ge::ComputeGraph::Vistor<ge::NodePtr> node_ptrs = compute_graph_ptr->GetAllNodes();

              buffer_fusion_pass_base_ptr->SetName(fusion_pass_name);
              vector<BufferFusionPattern*> patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
              for (auto pattern : patterns) {
                std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
                vector<ge::NodePtr> matmulNodes;
                for (auto i : node_ptrs) {
                  auto opDesc = i->GetOpDesc();
                  if (opDesc->GetType() == "Matmul") {
                    matmulNodes.push_back(i);
                  }
                }

                BufferFusionMapping mapping;
                for (auto i : desc) {
                  if (i->desc_name == "matmul") {
                    mapping[i] = matmulNodes;
                  }
                }

                vector<ge::NodePtr> fusion_nodes;
                InputSplitInfo input_split_info;
                vector<int64_t> axis = {0};
                int64_t idx = 0;
                vector<int64_t> overlap = {-1};
                input_split_info.Initialize();
                input_split_info.SetAxis(axis);
                input_split_info.SetIndex(idx);
                input_split_info.SetHeadOverLap(overlap);
                input_split_info.SetTailOverLap(overlap);
                OutputSplitInfo output_split_info;
                output_split_info.Initialize();
                output_split_info.SetAxis(axis);
                output_split_info.SetIndex(idx);
                AxisSplitMap split_map;
                split_map.Initialize();
                split_map.AddInputSplitInfo(input_split_info);
                split_map.AddOutputSplitInfo(output_split_info);
                vector<AxisSplitMap> split_map_vec = {split_map};
                SetSplitMapMainNode(split_map_vec, matmulNodes, "Matmul");
                buffer_fusion_pass_base_ptr->GetFusionNodes(mapping, fusion_nodes);
                buffer_fusion_pass_base_ptr->SetSplitInfo(mapping, fusion_nodes);
              }
              return SUCCESS;
          }
      }

      return FAILED;

  }
}

TEST_F(matmul_transdata_ub_fusion_test, matmul_transdata_ub_fusion_test_1) {
    ge::Graph graph("matmul_transdata_ub_fusion_test_1");

    DESC_DATA(data_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_a, ge::GeShape({64, 32, 16, 16}), FORMAT_FRACTAL_NZ, ge::GeShape({64, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(trans_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({256, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(matmul_y, ge::GeShape({512, 4096}), FORMAT_ND, ge::GeShape({256, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_c, ge::GeShape({512, 1024}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr matmul = std::make_shared<ge::OpDesc>("matmul", "Matmul");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    trans_b->AddInputDesc(desc_data_b);
    trans_b->AddOutputDesc(desc_trans_b);
    matmul->AddInputDesc("x1", desc_data_a);
    matmul->AddInputDesc("x2", desc_trans_b);
    matmul->AddOutputDesc(desc_matmul_y);
    trans_c->AddInputDesc(desc_matmul_y);
    trans_c->AddOutputDesc(desc_trans_c);
    netoutput->AddInputDesc(desc_trans_c);


    ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "dst_format", "ND");
    ge::AttrUtils::SetBool(matmul, "adj_x1", false);
    ge::AttrUtils::SetBool(matmul, "adj_x2", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr matmul_node = compute_graph_ptr->AddNode(matmul);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("MatmulTransdataFusionPass",
                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                         compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}