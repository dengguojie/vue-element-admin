#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/conv/conv2d_transdata_fusion_pass.h"
#include "common/util/platform_info.h"
#undef protected
#undef private

using namespace ge;

class ConvBinaryTransdataFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Conv binary transdata fusion pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv binary transdata fusion pass TearDown" << std::endl;
  }
};

TEST_F(ConvBinaryTransdataFusionTest, ConvBinaryTransdataFusionTest_1) {
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr transdata_pre = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr transdata_post = std::make_shared<OpDesc>("transdata", "TransData");

  vector<int64_t> dim_transdata1_in({-1, -1, -1, -1});
  GeShape shape_transdata1_in(dim_transdata1_in);
  vector<std::pair<int64_t,int64_t>> range_transdata1_in = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  GeTensorDesc desc_transdata1_in(shape_transdata1_in);
  desc_transdata1_in.SetOriginFormat(FORMAT_NCHW);
  desc_transdata1_in.SetOriginDataType(DT_FLOAT16);
  desc_transdata1_in.SetOriginShape(shape_transdata1_in);
  desc_transdata1_in.SetFormat(FORMAT_NCHW);
  desc_transdata1_in.SetDataType(DT_FLOAT16);
  desc_transdata1_in.SetShapeRange(range_transdata1_in);

  vector<int64_t> dim_transdata1_out({-1, -1, -1, -1, 16});
  GeShape shape_transdata1_out(dim_transdata1_out);
  vector<std::pair<int64_t,int64_t>> range_transdata1_out = {{1, -1}, {1, -1}, {1, -1}, {1, -1}, {16, 16}};
  GeTensorDesc desc_transdata1_out(shape_transdata1_out);
  desc_transdata1_out.SetFormat(FORMAT_NC1HWC0);
  desc_transdata1_out.SetDataType(DT_FLOAT16);
  desc_transdata1_out.SetShapeRange(range_transdata1_out);
  desc_transdata1_out.SetOriginShape(shape_transdata1_in);

  transdata_pre->AddInputDesc("src", desc_transdata1_in);
  transdata_pre->AddOutputDesc("dst", desc_transdata1_out);

  vector<int64_t> dim_filter({9, 1, 16, 16});
  GeShape shape_filter(dim_filter);
  vector<int64_t> dim_filter_ori({1, 16, 3, 3});
  GeShape shape_filter_ori(dim_filter_ori);
  GeTensorDesc filter_desc(shape_filter);
  filter_desc.SetOriginFormat(FORMAT_FRACTAL_Z);
  filter_desc.SetOriginDataType(DT_FLOAT16);
  filter_desc.SetOriginShape(shape_filter_ori);
  filter_desc.SetFormat(FORMAT_NCHW);
  filter_desc.SetDataType(DT_FLOAT16);

  vector<int64_t> dim_transdata2_in({-1, -1, -1, -1, 16});
  GeShape shape_transdata2_in(dim_transdata2_in);
  vector<std::pair<int64_t,int64_t>> range_transdata2_in = {{1, -1}, {1, -1}, {1, -1}, {1, -1}, {16, 16}};
  GeTensorDesc desc_transdata2_in(shape_transdata2_in);
  desc_transdata2_in.SetFormat(FORMAT_NC1HWC0);
  desc_transdata2_in.SetDataType(DT_FLOAT16);
  desc_transdata2_in.SetShapeRange(range_transdata2_in);

  conv_op->AddInputDesc("x", desc_transdata1_out);
  conv_op->AddInputDesc("filter", filter_desc);
  conv_op->AddOutputDesc(desc_transdata2_in);
  ge::AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});

  vector<int64_t> dim_transdata2_out({-1, -1, -1, -1});
  GeShape shape_transdata2_out(dim_transdata2_out);
  vector<std::pair<int64_t,int64_t>> range_transdata2_out = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  GeTensorDesc desc_transdata2_out(shape_transdata2_out);
  desc_transdata2_out.SetFormat(FORMAT_NCHW);
  desc_transdata2_out.SetDataType(DT_FLOAT16);
  desc_transdata2_out.SetShapeRange(range_transdata2_out);

  transdata_post->AddInputDesc("src", desc_transdata2_in);
  transdata_post->AddOutputDesc("dst", desc_transdata2_out);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr transdata_pre_node = graph->AddNode(transdata_pre);
  NodePtr transdata_post_node = graph->AddNode(transdata_post);
  ge::GraphUtils::AddEdge(transdata_pre_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), transdata_post_node->GetInDataAnchor(0));

  fe::Conv2dTransDataFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionPattern *pattern;
  fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "Conv2dTransDataFusionPassPattern1", &pattern);

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "transdata1") {
      vector<ge::NodePtr> nodes_transdata1;
      nodes_transdata1.push_back(transdata_pre_node);
      mapping[desc] = nodes_transdata1;
    } else if (desc->desc_name == "Convolution") {
      vector<ge::NodePtr> nodes_conv;
      nodes_conv.push_back(conv_node);
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "transdata2") {
      vector<ge::NodePtr> nodes_transdata2;
      nodes_transdata2.push_back(transdata_post_node);
      mapping[desc] = nodes_transdata2;
    }
  }

  vector<NodePtr> fusion_nodes;
  fe::Status status = fusion_pass.GetFusionNodes(mapping, fusion_nodes);
}

TEST_F(ConvBinaryTransdataFusionTest, ConvBinaryTransdataFusionTest_2) {
  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  setenv("ASCEND_OPP_PATH", "/", 1);

  fe::Conv2dTransDataFusionPass fusion_pass;
  fusion_pass.CheckBinaryReuse();

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}