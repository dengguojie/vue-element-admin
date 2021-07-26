#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_norm_ops.h"
#include "matrix_calculation_ops.h"
#include "common/lx_fusion_func.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/matmul/batch_matmul_dropout_do_mask_v3_d_ub_fusion.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class matmul_dropout_do_mask_v3_ub_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "matmul_dropout_do_mask_v3_ub_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "matmul_dropout_do_mask_v3_ub_fusion_test TearDown" << std::endl;
    }
};

namespace fe {
  static Status RunBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
    ge::ComputeGraphPtr& compute_graph_ptr) {
      std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
          BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
      const auto &iter = createFns.find(fusionPassName);
      if (iter != createFns.end()) {
          if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto BufferFusionPassBasePtr = std::unique_ptr<BatchMatmulDropOutDoMaskV3DFusionPass>(
                      dynamic_cast<BatchMatmulDropOutDoMaskV3DFusionPass *>(iter->second()));
              if (BufferFusionPassBasePtr == nullptr) {
                  return FAILED;
              }
              ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = compute_graph_ptr->GetAllNodes();
              std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

              BufferFusionPassBasePtr->SetName(fusionPassName);
              vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();
              for (auto pattern : patterns) {
                std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
                vector<ge::NodePtr> elemNodes;
                vector<ge::NodePtr> elemNode1;
                vector<ge::NodePtr> matmulNodes;
                for (auto i : NodePtrs) {
                  auto opDesc = i->GetOpDesc();
                  if (opDesc->GetType() == "BatchMatMul") {
                    matmulNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "DropOutDoMaskV3D") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({16, 512, 4096}));
                    opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({16, 512, 4096}));
                    elemNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "Add") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({16, 512, 4096}));
                    opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({16, 512, 4096}));
                    elemNode1.push_back(i);
                  }
                }

                BufferFusionMapping mapping;
                for (auto i : desc) {
                  if (i->desc_name == "batch_matmul") {
                    mapping[i] = matmulNodes;
                  }
                  if (i->desc_name == "dropout_do_mask_v3_d") {
                    mapping[i] = elemNodes;
                  }
                  if (i->desc_name == "add") {
                    mapping[i] = elemNode1;
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
                SetSplitMapMainNode(split_map_vec, matmulNodes, "BatchMatMul");
                BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
                BufferFusionPassBasePtr->SetSplitInfo(mapping, fusion_nodes);
              }
              return SUCCESS;
          }
      }

      return FAILED;

  }
}
TEST_F(matmul_dropout_do_mask_v3_ub_fusion_test, matmul_dropout_do_mask_v3_ub_fusion_test_1) {
    ge::Graph graph("matmul_dropout_do_mask_v3_ub_fusion_test_1");

    ge::TensorDesc a_desc(ge::Shape({16, 512, 1024}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_a = op::Data("data_a");
    data_a.update_input_desc_x(a_desc);
    data_a.update_output_desc_y(a_desc);

    ge::TensorDesc b_desc(ge::Shape({16, 1024, 4096}), ge::FORMAT_NHWC, DT_FLOAT16);
    auto data_b = op::Data("data_b");
    data_b.update_input_desc_x(b_desc);
    data_b.update_output_desc_y(b_desc);

    auto batch_matmul_op = op::BatchMatMul("BatchMatMul")
        .set_input_x1(data_a)
        .set_input_x2(data_b)
        .set_attr_adj_x1(false)
        .set_attr_adj_x2(false);

    ge::TensorDesc dropout_mask_desc(ge::Shape({16, 512, 4096}), ge::FORMAT_NHWC, ge::DT_UINT8);
    auto data_dropout_mask = op::Data("data_dropout");
    data_dropout_mask.update_input_desc_x(dropout_mask_desc);
    data_dropout_mask.update_output_desc_y(dropout_mask_desc);

    auto dropout_op = op::DropOutDoMaskV3D("DropOutDoMaskV3D")
        .set_input_x(batch_matmul_op)
        .set_input_mask(data_dropout_mask)
        .set_attr_keep_prob(0.5);

    ge::TensorDesc add_desc(ge::Shape({16, 512, 4096}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_add = op::Data("data_add");
    data_add.update_input_desc_x(add_desc);
    data_add.update_output_desc_y(add_desc);

    auto add_op = op::Add("Add")
        .set_input_x1(dropout_op)
        .set_input_x2(data_add);

    std::vector<Operator> inputs{data_a, data_b, data_dropout_mask, data_add};
    std::vector<Operator> outputs{add_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("BatchMatmulDropOutDoMaskV3DFusionPass",
                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                         compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}


