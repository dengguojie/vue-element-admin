#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/conv3d/conv3d_elemwise_pass.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"

using namespace ge;
using namespace op;

class conv3d_elemwise_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_elemwise_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_elemwise_fusion_test TearDown" << std::endl;
    }
};

namespace fe {
  static Status RunConv3dBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
                                                   ge::ComputeGraph &computeGraph) {
      std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
              BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
      const auto &iter = createFns.find(fusionPassName);
      if (iter != createFns.end()) {
          if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto BufferFusionPassBasePtr = std::unique_ptr<TbeConv3dElemwisePass>(
                      dynamic_cast<TbeConv3dElemwisePass *>(iter->second()));
              if (BufferFusionPassBasePtr == nullptr) {
                  return FAILED;
              }
              BufferFusionPassBasePtr->SetName(fusionPassName);
              vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();

              std::vector<BufferFusionOpDesc *> desc = patterns[0]->GetOpDescs();

              ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = computeGraph.GetAllNodes();
              std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

              vector<ge::NodePtr> elemNodes;
              vector<ge::NodePtr> conv3dNodes;
              for (auto i : NodePtrs) {
                auto opDesc = i->GetOpDesc();
                if (opDesc->GetType() == "Conv3D") {
                  conv3dNodes.push_back(i);
                }
                if (opDesc->GetType() == "Mul") {
                  opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({1, 32, 1, 240, 352, 16}));
                  opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({1, 32, 1, 240, 352, 16}));
                  elemNodes.push_back(i);
                }
              }

              BufferFusionMapping mapping;
              for (auto i : desc) {
                if (i->desc_name == "conv3d") {
                  mapping[i] = conv3dNodes;
                }
                if (i->desc_name == "elemwise") {
                  mapping[i] = elemNodes;
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
              SetSplitMapMainNode(split_map_vec, conv3dNodes, "Conv3dBackpropOp");
              BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
              BufferFusionPassBasePtr->SetSplitInfo(mapping, fusion_nodes);

              return SUCCESS;
          }
      }

      return FAILED;

  }
}
TEST_F(conv3d_elemwise_fusion_test, conv3d_elemwise_fusion_test_1) {
    ge::Graph graph("conv3d_elemwise_fusion_test_1");

    auto x_shape = vector<int64_t>({1, 32, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({3,3,3,16,16}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d = op::Conv3D("conv3d")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 1, 1, 1})
        .set_attr_pads({1, 1, 1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 32, 240, 352, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    auto mul = op::Mul("mul")
        .set_input_x1(conv3d)
        .set_input_x2(data_x2);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{mul};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, *compute_graph_ptr);

    bool find_mul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            find_mul = true;
        }
    }
    EXPECT_EQ(find_mul, true);
}
