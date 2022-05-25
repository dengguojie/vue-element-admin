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
#include "buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_dequant_sigmoid_mul_add_pass.h"
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace fe;
using namespace op;

class ConvDequantSigmoidMulAdd : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ConvDequantSigmoidMulAdd SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ConvDequantSigmoidMulAdd TearDown" << std::endl;
    }
};

namespace fe {
  static Status RunConv2dBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
                                                   ge::ComputeGraph &computeGraph) {
      std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
              BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
      const auto &iter = createFns.find(fusionPassName);
      if (iter != createFns.end()) {
          if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto BufferFusionPassBasePtr = std::unique_ptr<ConvDequantSigmoidMulAddFusionPass>(
                      dynamic_cast<ConvDequantSigmoidMulAddFusionPass *>(iter->second()));
              if (BufferFusionPassBasePtr == nullptr) {
                  return FAILED;
              }
              BufferFusionPassBasePtr->SetName(fusionPassName);
              vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();

              std::vector<BufferFusionOpDesc *> desc = patterns[0]->GetOpDescs();

              ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = computeGraph.GetAllNodes();
              std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

              vector<ge::NodePtr> elemNodes;
              vector<ge::NodePtr> conv2dNodes;
              for (auto i : NodePtrs) {
                auto opDesc = i->GetOpDesc();
                if (opDesc->GetType() == "Conv2D") {
                  conv2dNodes.push_back(i);
                }
                if (opDesc->GetType() == "Sigmoid" || opDesc->GetType() == "Relu") {
                  elemNodes.push_back(i);
                }
                if (opDesc->GetType() == "Mul" || opDesc->GetType() == "Div") {
                  elemNodes.push_back(i);
                }
                if (opDesc->GetType() == "Add" || opDesc->GetType() == "Sub") {
                  elemNodes.push_back(i);
                }
              }

              BufferFusionMapping mapping;
              for (auto i : desc) {
                if (i->desc_name == "convolution" || i->desc_name == "conv2d") {
                  mapping[i] = conv2dNodes;
                }
                if (i->desc_name == "sigmoid") {
                  mapping[i] = elemNodes;
                }
                if (i->desc_name == "mul") {
                  mapping[i] = elemNodes;
                }
                if (i->desc_name == "add") {
                  mapping[i] = elemNodes;
                }
              }
              vector<ge::NodePtr> fusion_nodes;
              InputSplitInfo input_split_info;
              vector<int64_t> axis = {0};
              int64_t idx = 0;
              vector<int64_t> overlap = {1};
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
              SetSplitMapMainNode(split_map_vec, conv2dNodes, "Conv2d");
              BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
              BufferFusionPassBasePtr->SetSplitInfo(mapping, fusion_nodes);
              return SUCCESS;
          }
      }

      return FAILED;

  }
}
TEST_F(ConvDequantSigmoidMulAdd, ConvDequantSigmoidMulAdd_1) {
    ge::Graph graph("ConvDequantSigmoidMulAdd_1");

    auto x_shape = vector<int64_t>({1, 8, 40, 40, 32});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({256, 256, 3, 3}), FORMAT_FRACTAL_Z, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv2d = op::Conv2D("convolution")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 1, 1})
        .set_attr_pads({1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NC1HWC0");

    TensorDesc conv2d_input_desc_x(ge::Shape(), FORMAT_NC1HWC0, DT_FLOAT16);
    TensorDesc conv2d_input_desc_filter(ge::Shape(), FORMAT_FRACTAL_Z, DT_FLOAT16);
    TensorDesc conv2d_output_desc_y(ge::Shape(), FORMAT_NC1HWC0, DT_FLOAT16);
    conv2d.update_input_desc_x(conv2d_input_desc_x);
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);
    conv2d.update_output_desc_y(conv2d_output_desc_y);


    auto sigmoid = op::Sigmoid("sigmoid").set_input_x(conv2d);
    auto x2_shape = vector<int64_t>({1, 16, 40, 40, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x2.update_input_desc_x(x2_desc);
    data_x2.update_output_desc_y(x2_desc);

    auto mul = op::Mul("mul")
        .set_input_x1(conv2d)
        .set_input_x2(sigmoid);

    auto add = op::Add("add");
    ge::TensorDesc add_input_desc_x1_1(ge::Shape({1, 16, 40, 40, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    ge::TensorDesc add_input_desc_x2_1(ge::Shape({1, 16, 40, 40, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    ge::TensorDesc add_output_desc_y_1(ge::Shape({1, 16, 40, 40, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    add.update_input_desc_x1(add_input_desc_x1_1);
    add.update_input_desc_x2(add_input_desc_x2_1);
    add.update_output_desc_y(add_output_desc_y_1);
    add.set_input_x1(mul);
    add.set_input_x2(data_x2);
    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{add};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    RunConv2dBufferFusionPass("TbeConvDequantSigmoidMulAddFusionPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, *compute_graph_ptr);

    bool find_mul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            find_mul = true;
        }
    }
    EXPECT_EQ(find_mul, true);
}

TEST_F(ConvDequantSigmoidMulAdd, ConvDequantSigmoidMulAdd_2) {
    ge::Graph graph("ConvDequantSigmoidMulAdd_2");

    auto x_shape = vector<int64_t>({1, 8, 40, 40, 32});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({256, 256, 3, 3}), FORMAT_FRACTAL_Z, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv2d = op::Conv2D("conv2d")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 1, 1})
        .set_attr_pads({1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NC1HWC0");

    TensorDesc conv2d_input_desc_x(ge::Shape(), FORMAT_NC1HWC0, DT_FLOAT16);
    TensorDesc conv2d_input_desc_filter(ge::Shape(), FORMAT_FRACTAL_Z, DT_FLOAT16);
    TensorDesc conv2d_output_desc_y(ge::Shape(), FORMAT_NC1HWC0, DT_FLOAT16);
    conv2d.update_input_desc_x(conv2d_input_desc_x);
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);
    conv2d.update_output_desc_y(conv2d_output_desc_y);


    auto relu = op::Relu("sigmoid").set_input_x(conv2d);
    auto x2_shape = vector<int64_t>({1, 16, 40, 40, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x2.update_input_desc_x(x2_desc);
    data_x2.update_output_desc_y(x2_desc);

    auto div = op::Div("mul")
        .set_input_x1(conv2d)
        .set_input_x2(relu);

    auto sub = op::Sub("add")
        .set_input_x1(div)
        .set_input_x2(data_x2);
    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{sub};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    RunConv2dBufferFusionPass("TbeConvDequantSigmoidMulAddFusionPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, *compute_graph_ptr);

    bool find_mul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            find_mul = true;
        }
    }
    EXPECT_EQ(find_mul, false);
}

TEST_F(ConvDequantSigmoidMulAdd, ConvDequantSigmoidMulAdd_3)  {
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

  fe::ConvDequantSigmoidMulAddFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc fusion_desc;
  fusion_desc.desc_name = "elemwise";
  const fe::BufferFusionOpDesc *fusion_desc_ptr = &fusion_desc;
  fe::BufferFusionMapping mapping;
  vector<NodePtr> elemwise_nodes = {relu_node, sqrt_node, relu6_node};
  mapping.emplace(fusion_desc_ptr, elemwise_nodes);
  vector<NodePtr> fusion_nodes;

  fusion_pass.GetFusionNodes(mapping, fusion_nodes);
  fusion_pass.SetSplitInfo(mapping, fusion_nodes);
  fusion_desc_ptr = nullptr;
}
