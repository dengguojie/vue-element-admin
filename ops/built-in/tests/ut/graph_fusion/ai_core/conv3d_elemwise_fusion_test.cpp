#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/conv3d/conv3d_elemwise_pass.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "quantize_ops.h"
#include "transformation_ops.h"

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
                                          ge::ComputeGraph &computeGraph, vector<ge::NodePtr> &fusion_nodes) {
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
              vector<ge::NodePtr> dequant_nodes;
              vector<ge::NodePtr> requant_nodes;
              for (auto i : NodePtrs) {
                auto opDesc = i->GetOpDesc();
                if (opDesc->GetType() == "Conv3D") {
                  conv3dNodes.push_back(i);
                }
                if (opDesc->GetType() == "Mul" or opDesc->GetType() == "Add") {
                  elemNodes.push_back(i);
                } else if (opDesc->GetType() == "Relu") {
                  elemNodes.push_back(i);
                }
                if (opDesc->GetType() == "AscendDequant") {
                  dequant_nodes.push_back(i);
                }
                if (opDesc->GetType() == "AscendRequant") {
                  requant_nodes.push_back(i);
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
                if (i->desc_name == "dequant") {
                  mapping[i] = dequant_nodes;
                }
                if (i->desc_name == "requant") {
                  mapping[i] = requant_nodes;
                }
              }
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
              SetSplitMapMainNode(split_map_vec, conv3dNodes, "Conv3dOp");
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

    auto x_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({27,1,16,16}), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
    filter_desc.SetOriginShape(ge::Shape({3, 3, 3, 16, 16}));
    filter_desc.SetOriginFormat(ge::FORMAT_DHWCN);
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

    TensorDesc conv3d_output_desc_y(ge::Shape({1, 32, 1, 240, 352, 16}), FORMAT_NDC1HWC0, DT_FLOAT16);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NDHWC);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x2_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x2_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x2 = op::Data("data_x2");
    data_x2.update_input_desc_x(x2_desc);
    data_x2.update_output_desc_y(x2_desc);

    auto mul = op::Mul("mul")
        .set_input_x1(conv3d)
        .set_input_x2(data_x2);
    mul.update_input_desc_x1(conv3d_output_desc_y);
    mul.update_input_desc_x2(x2_desc);
    mul.update_output_desc_y(conv3d_output_desc_y);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{mul};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_mul fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_mul = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_mul fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "Mul") {
          find_mul = true;
      }
    }
    EXPECT_EQ(find_mul, true);
}


TEST_F(conv3d_elemwise_fusion_test, conv3d_add_relu) {
    ge::Graph graph("conv3d_add_relu");

    auto x_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({27,1,16,16}), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
    filter_desc.SetOriginShape(ge::Shape({3, 3, 3, 16, 16}));
    filter_desc.SetOriginFormat(ge::FORMAT_DHWCN);
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

    TensorDesc conv3d_output_desc_y(ge::Shape({1, 32, 1, 240, 352, 16}), FORMAT_NDC1HWC0, DT_FLOAT16);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NDHWC);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x2_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x2_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x2 = op::Data("data_x2");
    data_x2.update_input_desc_x(x2_desc);
    data_x2.update_output_desc_y(x2_desc);

    auto add = op::Add("add")
        .set_input_x1(conv3d)
        .set_input_x2(data_x2);
    add.update_input_desc_x1(conv3d_output_desc_y);
    add.update_input_desc_x2(x2_desc);
    add.update_output_desc_y(conv3d_output_desc_y);
    auto relu = op::Relu("relu")
        .set_input_x(add);
    relu.update_input_desc_x(conv3d_output_desc_y);
    relu.update_output_desc_y(conv3d_output_desc_y);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_add_relu fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_add = false;
    bool find_relu = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_add_relu fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "Add") {
          find_add = true;
      } else if (node->GetType() == "Relu") {
          find_relu = true;
      }
    }
    EXPECT_EQ(find_add, true);
    EXPECT_EQ(find_relu, true);
}


TEST_F(conv3d_elemwise_fusion_test, conv3d_relu6) {
    ge::Graph graph("conv3d_relu6");

    auto x_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({27,1,16,16}), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
    filter_desc.SetOriginShape(ge::Shape({3, 3, 3, 16, 16}));
    filter_desc.SetOriginFormat(ge::FORMAT_DHWCN);
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

    TensorDesc conv3d_output_desc_y(ge::Shape({1, 32, 1, 240, 352, 16}), FORMAT_NDC1HWC0, DT_FLOAT16);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NDHWC);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto relu = op::Relu6("relu6")
        .set_input_x(conv3d);
    relu.update_input_desc_x(conv3d_output_desc_y);
    relu.update_output_desc_y(conv3d_output_desc_y);

    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_relu6 fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_relu = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_relu6 fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "Relu6") {
          find_relu = true;
      }
    }
    EXPECT_EQ(find_relu, false);
}


TEST_F(conv3d_elemwise_fusion_test, conv3d_relu) {
    ge::Graph graph("conv3d_relu");

    auto x_shape = vector<int64_t>({1, 32, 1, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x_desc.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    x_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({27,1,16,16}), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
    filter_desc.SetOriginShape(ge::Shape({3, 3, 3, 16, 16}));
    filter_desc.SetOriginFormat(ge::FORMAT_DHWCN);
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

    TensorDesc conv3d_output_desc_y(ge::Shape({1, 32, 1, 240, 352, 16}), FORMAT_NDC1HWC0, DT_FLOAT16);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NDHWC);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto relu = op::Relu("relu")
        .set_input_x(conv3d);
    relu.update_input_desc_x(conv3d_output_desc_y);
    relu.update_output_desc_y(conv3d_output_desc_y);

    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_relu fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_relu = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_relu fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "Relu") {
          find_relu = true;
      }
    }
    EXPECT_EQ(find_relu, true);

    for (auto node: compute_graph_ptr->GetAllNodes()) {
      std::cout << "conv3d_relu graph_node:" << node->GetType().c_str() << std::endl;
    }
}


TEST_F(conv3d_elemwise_fusion_test, conv3d_dyn) {
    ge::Graph graph("conv3d_dyn");

    auto x_shape = vector<int64_t>({1, 32, 1, -1, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    x_desc.SetOriginShape(ge::Shape({1, 32, -1, 352, 16}));
    x_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({27, 1,16,16}), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
    filter_desc.SetOriginShape(ge::Shape({3, 3, 3, 16, 16}));
    filter_desc.SetOriginFormat(ge::FORMAT_DHWCN);
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

    TensorDesc conv3d_output_desc_y(ge::Shape({1, 32, 1, 240, 352, 16}), FORMAT_NDC1HWC0, DT_FLOAT16);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({1, 32, 240, 352, 16}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NDHWC);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto relu = op::Relu("relu")
        .set_input_x(conv3d);
    relu.update_input_desc_x(conv3d_output_desc_y);
    relu.update_output_desc_y(conv3d_output_desc_y);

    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{relu};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_dyn fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_relu = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_dyn fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "Relu") {
          find_relu = true;
      }
    }
    EXPECT_EQ(find_relu, false);
}


TEST_F(conv3d_elemwise_fusion_test, conv3d_dequant_fusion_test) {
    ge::Graph graph("conv3d_dequant_fusion_test");

    auto x_shape = vector<int64_t>({3, 98, 1, 116, 67, 32});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_INT8);
    x_desc.SetOriginShape(ge::Shape({3, 26, 98, 116, 67}));
    x_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({104, 5, 16, 32}), FORMAT_FRACTAL_Z_3D, DT_INT8);
    filter_desc.SetOriginShape(ge::Shape({76, 26, 26, 1, 4}));
    filter_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d = op::Conv3D("conv3d")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 6, 22, 15})
        .set_attr_pads({12, 12, 0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NCDHW");
    TensorDesc conv3d_output_desc_y(ge::Shape({3, 17, 5, 6, 5, 16}), FORMAT_NDC1HWC0, DT_INT32);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({3, 76, 17, 6, 5}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NCDHW);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    TensorDesc deq_desc(ge::Shape({76}), FORMAT_NDC1HWC0, DT_FLOAT);
    deq_desc.SetOriginShape(ge::Shape({76}));
    deq_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    float deqScale = 1.0;
    Tensor deq_scale_tensor(deq_desc, reinterpret_cast<uint8_t*>(&deqScale), sizeof(float));
    auto deq_scale = op::Const("deq_scale").set_attr_value(deq_scale_tensor);

    auto dequant_op = op::AscendDequant("dequant");
    dequant_op.set_input_x(conv3d);
    dequant_op.set_input_deq_scale(deq_scale);

    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{dequant_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_dequant fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_dequant = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_dequant fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "AscendDequant") {
          find_dequant = true;
      }
    }
    EXPECT_EQ(find_dequant, false);
}

TEST_F(conv3d_elemwise_fusion_test, conv3d_requant_fusion_test) {
    ge::Graph graph("conv3d_requant_fusion_test");

    auto x_shape = vector<int64_t>({3, 98, 1, 116, 67, 32});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_INT8);
    x_desc.SetOriginShape(ge::Shape({3, 26, 98, 116, 67}));
    x_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({104, 5, 16, 32}), FORMAT_FRACTAL_Z_3D, DT_INT8);
    filter_desc.SetOriginShape(ge::Shape({76, 26, 26, 1, 4}));
    filter_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d = op::Conv3D("conv3d")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 6, 22, 15})
        .set_attr_pads({12, 12, 0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NCDHW");
    TensorDesc conv3d_output_desc_y(ge::Shape({3, 17, 5, 6, 5, 16}), FORMAT_NDC1HWC0, DT_INT32);
    conv3d_output_desc_y.SetOriginShape(ge::Shape({3, 76, 17, 6, 5}));
    conv3d_output_desc_y.SetOriginFormat(ge::FORMAT_NCDHW);
    conv3d.update_input_desc_x(x_desc);
    conv3d.update_input_desc_filter(filter_desc);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    TensorDesc req_desc(ge::Shape({76}), FORMAT_NDC1HWC0, DT_UINT64);
    req_desc.SetOriginShape(ge::Shape({76}));
    req_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    Tensor req_scale_tensor(req_desc);
    auto req_scale = op::Const().set_attr_value(req_scale_tensor);

    auto requant_op = op::AscendRequant("requant");
    requant_op.set_input_x(conv3d);
    requant_op.set_input_req_scale(req_scale);

    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{requant_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    vector<ge::NodePtr> fusion_nodes;
    Status res = RunConv3dBufferFusionPass("TbeConv3dElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           *compute_graph_ptr, fusion_nodes);
    EXPECT_EQ(res, fe::SUCCESS);
    std::cout << "conv3d_requant fusion_node:" << fusion_nodes.size() << std::endl;
    bool find_requant = false;
    for (auto node: fusion_nodes) {
      std::cout << "conv3d_requant fusion_node:" << node->GetType().c_str() << std::endl;
      if (node->GetType() == "AscendRequant") {
          find_requant = true;
      }
    }
    EXPECT_EQ(find_requant, false);
}
