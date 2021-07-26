#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "common/lx_fusion_func.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/conv3d_backprop_input/conv3d_bp_input_elemwise_pass.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class conv3d_bp_input_elemwise_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_bp_input_elemwise_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_bp_input_elemwise_fusion_test TearDown" << std::endl;
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
              auto BufferFusionPassBasePtr = std::unique_ptr<TbeConv3dDxElemwisePass>(
                      dynamic_cast<TbeConv3dDxElemwisePass *>(iter->second()));
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
                vector<ge::NodePtr> conv3dNodes;
                for (auto i : NodePtrs) {
                  auto opDesc = i->GetOpDesc();
                  if (opDesc->GetType() == "Conv3DBackpropInputD") {
                    conv3dNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "AddN") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    elemNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "LeakyReluGrad") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    elemNode1.push_back(i);
                  }
                  if (opDesc->GetType() == "Mul") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    opDesc->MutableInputDesc(1)->SetShape(ge::GeShape({1, 4, 8, 30, 44, 16}));
                    elemNode1.push_back(i);
                  }
                }

                BufferFusionMapping mapping;
                for (auto i : desc) {
                  if (i->desc_name == "conv3dbackpropinput") {
                    mapping[i] = conv3dNodes;
                  }
                  if (i->desc_name == "elemwise") {
                    mapping[i] = elemNodes;
                  }
                  if (i->desc_name == "elemwise1") {
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
                SetSplitMapMainNode(split_map_vec, conv3dNodes, "Conv3dBackpropOp");
                BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
                BufferFusionPassBasePtr->SetSplitInfo(mapping, fusion_nodes);
              }
              return SUCCESS;
          }
      }

      return FAILED;

  }
}
TEST_F(conv3d_bp_input_elemwise_fusion_test, conv3d_bp_input_elemwise_fusion_test_1) {
    ge::Graph graph("conv3d_bp_input_elemwise_fusion_test_1");

    auto x_shape = vector<int64_t>({1, 2, 15, 22, 256});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({2, 2, 2, 128, 256}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d_dx = op::Conv3DBackpropInputD("conv3d_backprop_input_d")
        .set_input_out_backprop(data_x)
        .set_input_filter(data_filter)
        .set_attr_input_size({1, 4, 30, 44, 128})
        .set_attr_strides({1, 2, 2, 2, 1})
        .set_attr_pads({1, 1, 1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_out_backprop(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d_dx.update_input_desc_out_backprop(conv3d_input_desc_out_backprop);
    conv3d_dx.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d_dx.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 4, 30, 44, 128});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    auto relu = op::LeakyReluGrad("leaky_relu_grad")
        .set_input_gradients(conv3d_dx)
        .set_input_features(data_x2);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("TbeConv3dDxElemwisePass",
                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                         compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(conv3d_bp_input_elemwise_fusion_test, conv3d_bp_input_elemwise_fusion_test_2) {
    ge::Graph graph("conv3d_bp_input_elemwise_fusion_test_2");

    auto x_shape = vector<int64_t>({1, 2, 15, 22, 256});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({2, 2, 2, 128, 256}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d_dx = op::Conv3DBackpropInputD("conv3d_backprop_input_d")
        .set_input_out_backprop(data_x)
        .set_input_filter(data_filter)
        .set_attr_input_size({1, 4, 30, 44, 128})
        .set_attr_strides({1, 2, 2, 2, 1})
        .set_attr_pads({1, 1, 1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_out_backprop(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d_dx.update_input_desc_out_backprop(conv3d_input_desc_out_backprop);
    conv3d_dx.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d_dx.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 4, 30, 44, 128});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    auto relu = op::Mul("mul")
        .set_input_x1(conv3d_dx)
        .set_input_x2(data_x2);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("TbeConv3dDxElemwisePass",
                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                         compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}

