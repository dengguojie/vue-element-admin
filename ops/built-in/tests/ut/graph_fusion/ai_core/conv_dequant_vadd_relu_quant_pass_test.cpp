#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_dequant_vadd_relu_quant_pass.h"
#include "common/util/platform_info.h"
#include "fusion_pass_test_utils.h"
#undef protected
#undef private

using namespace ge;

class ConvDequantVaddReluQuant : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Conv dequant vadd relu quant pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv dequant vadd relu quant pass TearDown" << std::endl;
  }
};

TEST_F(ConvDequantVaddReluQuant, ConvDequantVaddReluQuant_1) {
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

  fe::ConvDequantVaddReluQuantFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc fusion_desc;
  fusion_desc.desc_name = "elemwise";
  const fe::BufferFusionOpDesc *fusion_desc_ptr = &fusion_desc;
  fe::BufferFusionMapping mapping;
  vector<NodePtr> elemwise_nodes = {relu_node, sqrt_node, relu6_node};
  mapping.emplace(fusion_desc_ptr, elemwise_nodes);
  vector<NodePtr> fusion_nodes;

  fusion_pass.GetFusionNodes(mapping, fusion_nodes);
  bool read_select_flag = false;
  fusion_pass.SetMemoryReuse(mapping, read_select_flag);
  fusion_desc_ptr = nullptr;
}

TEST_F(ConvDequantVaddReluQuant, ConvDequantVaddReluQuant_2) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  platformInfo.str_info.short_soc_version = "Ascend310P";
  optiCompilationInfo.soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr dequant_op = std::make_shared<OpDesc>("dequant", "AscendDequant");
  OpDescPtr readselect_op = std::make_shared<OpDesc>("read_select", "ReadSelect");
  OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr quant_op = std::make_shared<OpDesc>("quant", "AscendQuant");

  //=============================conv====================================
  GeShape shape_in({1, 32, 16, 16});
  GeShape shape_in_5hd({1, 1, 16, 16, 32});
  GeTensorDesc desc_in(shape_in);
  desc_in.SetOriginFormat(FORMAT_NCHW);
  desc_in.SetOriginDataType(DT_INT8);
  desc_in.SetOriginShape(shape_in);
  desc_in.SetFormat(FORMAT_NC1HWC0);
  desc_in.SetDataType(DT_INT8);
  desc_in.SetShape(shape_in_5hd);

  GeShape shape_w({32, 32, 1, 1});
  GeShape shape_w_fracz({1, 2, 16, 32});
  GeTensorDesc desc_w(shape_w);
  desc_w.SetOriginFormat(FORMAT_NCHW);
  desc_w.SetOriginDataType(DT_INT8);
  desc_w.SetOriginShape(shape_w);
  desc_w.SetFormat(FORMAT_FRACTAL_Z);
  desc_w.SetDataType(DT_INT8);
  desc_w.SetShape(shape_w_fracz);

  GeShape shape_out({1, 32, 16, 16});
  GeShape shape_out_5hd({1, 2, 16, 16, 16});
  GeTensorDesc desc_out(shape_out);
  desc_out.SetOriginFormat(FORMAT_NCHW);
  desc_out.SetOriginDataType(DT_INT32);
  desc_out.SetOriginShape(shape_out);
  desc_out.SetFormat(FORMAT_NC1HWC0);
  desc_out.SetDataType(DT_INT32);
  desc_out.SetShape(shape_out_5hd);

  conv_op->AddInputDesc("x", desc_in);
  conv_op->AddInputDesc("filter", desc_w);
  conv_op->AddOutputDesc(desc_out);
  ge::AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});

  //=============================dequant====================================
  GeShape shape_dequant_out({1, 32, 16, 16});
  GeShape shape_dequant_out_5hd({1, 2, 16, 16, 16});
  GeTensorDesc tensor_desc_deqout(shape_dequant_out);
  tensor_desc_deqout.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_deqout.SetOriginDataType(DT_FLOAT16);
  tensor_desc_deqout.SetOriginShape(shape_dequant_out);
  tensor_desc_deqout.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_deqout.SetDataType(DT_FLOAT16);
  tensor_desc_deqout.SetShape(shape_dequant_out_5hd);

  GeShape shape_deqscale({32});
  GeShape shape_deqscale_5hd({1, 2, 1, 1, 16});
  GeTensorDesc tensor_desc_deqscale(shape_deqscale);
  tensor_desc_deqscale.SetOriginFormat(FORMAT_ND);
  tensor_desc_deqscale.SetOriginDataType(DT_FLOAT16);
  tensor_desc_deqscale.SetOriginShape(shape_deqscale);
  tensor_desc_deqscale.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_deqscale.SetDataType(DT_UINT64);
  tensor_desc_deqscale.SetShape(shape_deqscale_5hd);

  dequant_op->AddInputDesc("x", desc_out);
  dequant_op->AddInputDesc("deq_scale", tensor_desc_deqscale);
  dequant_op->AddOutputDesc(tensor_desc_deqout);

  //============================read_select===============================
  GeShape shape_readin({1, 32, 32, 16});
  GeShape shape_readin_5hd({1, 2, 32, 16, 16});
  GeTensorDesc tensor_desc_readin(shape_readin);
  tensor_desc_readin.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_readin.SetOriginDataType(DT_FLOAT16);
  tensor_desc_readin.SetOriginShape(shape_readin);
  tensor_desc_readin.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_readin.SetDataType(DT_FLOAT16);
  tensor_desc_readin.SetShape(shape_readin_5hd);

  GeShape shape_readout({1, 32, 16, 16});
  GeShape shape_readout_5hd({1, 2, 16, 16, 16});
  GeTensorDesc tensor_desc_readout(shape_readout);
  tensor_desc_readout.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_readout.SetOriginDataType(DT_FLOAT16);
  tensor_desc_readout.SetOriginShape(shape_readout);
  tensor_desc_readout.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_readout.SetDataType(DT_FLOAT16);
  tensor_desc_readout.SetShape(shape_readout_5hd);

  readselect_op->AddInputDesc("x", tensor_desc_readin);
  readselect_op->AddOutputDesc(tensor_desc_readout);

  //==========================Add======================================
  GeShape shape_add_other_in({1, 32, 16, 16});
  GeShape shape_add_other_in_5hd({1, 2, 16, 16, 16});
  GeTensorDesc tensor_desc_add_other_in(shape_add_other_in);
  tensor_desc_add_other_in.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_add_other_in.SetOriginDataType(DT_FLOAT16);
  tensor_desc_add_other_in.SetOriginShape(shape_add_other_in);
  tensor_desc_add_other_in.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_add_other_in.SetDataType(DT_FLOAT16);
  tensor_desc_add_other_in.SetShape(shape_add_other_in_5hd);

  add_op->AddInputDesc("x1", tensor_desc_readout);
  add_op->AddInputDesc("x2", tensor_desc_add_other_in);
  add_op->AddOutputDesc(tensor_desc_readout);

  //==========================relu===================================
  relu_op->AddInputDesc("x", tensor_desc_readout);
  relu_op->AddOutputDesc(tensor_desc_readout);

  //==========================quant===================================
  GeShape shape_quant_out({1, 32, 16, 16});
  GeShape shape_quant_out_5hd({1, 1, 16, 16, 32});
  GeTensorDesc tensor_desc_quant_out(shape_quant_out);
  tensor_desc_quant_out.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_quant_out.SetOriginDataType(DT_INT8);
  tensor_desc_quant_out.SetOriginShape(shape_quant_out);
  tensor_desc_quant_out.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_quant_out.SetDataType(DT_INT8);
  tensor_desc_quant_out.SetShape(shape_quant_out_5hd);

  quant_op->AddInputDesc("x", tensor_desc_readout);
  quant_op->AddOutputDesc(tensor_desc_quant_out);

  //=====================build graph=====================================
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr dequant_node = graph->AddNode(dequant_op);
  NodePtr readselect_node = graph->AddNode(readselect_op);
  NodePtr add_node = graph->AddNode(add_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr quant_node = graph->AddNode(quant_op);

  ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(readselect_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));

  fe::ConvDequantVaddReluQuantFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionPattern *pattern;
  fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvDequantVaddReluQuantFusion", &pattern);

  using namespace std;
  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      vector<ge::NodePtr> nodes_conv;
      nodes_conv.push_back(conv_node);
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "dequant") {
      vector<ge::NodePtr> nodes_dequant;
      nodes_dequant.push_back(dequant_node);
      mapping[desc] = nodes_dequant;
    } else if (desc->desc_name == "vadd") {
      vector<ge::NodePtr> nodes_vadd;
      nodes_vadd.push_back(add_node);
      mapping[desc] = nodes_vadd;
    } else if (desc->desc_name == "relu") {
      vector<ge::NodePtr> nodes_relu;
      nodes_relu.push_back(relu_node);
      mapping[desc] = nodes_relu;
    } else if (desc->desc_name == "quant") {
      vector<ge::NodePtr> nodes_quant;
      nodes_quant.push_back(quant_node);
      mapping[desc] = nodes_quant;
    }
  }

  vector<NodePtr> fusion_nodes;
  fe::Status status = fusion_pass.GetFusionNodes(mapping, fusion_nodes);
}
