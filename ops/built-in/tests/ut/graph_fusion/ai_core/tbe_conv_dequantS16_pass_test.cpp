#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "reduce_ops.h"
#include "nonlinear_fuc_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph_builder_utils.h"

using namespace ge;
using namespace op;

class TbeConvDequantS16FusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_dequant_add_mul_quant_pass_test SetUp" << std::endl;
    std::map<string, fe::BufferFusionPassRegistry::CreateFn> createFns =
        fe::BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type_fusion_pass);
    const auto &iter = createFns.find(name_fusion_pass);
    
    if (iter != createFns.end()) {
        ptr_buffer_fusion_pass_func =
            std::unique_ptr<fe::BufferFusionPassBase>(dynamic_cast<fe::BufferFusionPassBase *>(iter->second()));
        EXPECT_NE(ptr_buffer_fusion_pass_func, nullptr);
        ptr_buffer_fusion_pass_func->SetName(name_fusion_pass);
        patterns = ptr_buffer_fusion_pass_func->DefinePatterns();
    }

    EXPECT_NE(patterns.size(), 0);
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_dequant_add_mul_quant_pass_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfConvDequantS16(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};

vector<fe::BufferFusionPattern *> TbeConvDequantS16FusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> TbeConvDequantS16FusionTest::ptr_buffer_fusion_pass_func;
const string TbeConvDequantS16FusionTest::name_fusion_pass = "TbeConvDequantS16FusionPass";
const fe::BufferFusionPassType TbeConvDequantS16FusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping TbeConvDequantS16FusionTest::ConstructFusionMappingOfConvDequantS16(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_dequants16;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetName() == "AscendDequantS16") {
      nodes_dequants16.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvDequantS16Fusion", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "conv2d") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "dequants16") {
      mapping[desc] = nodes_dequants16;
    } else if (desc->desc_name == "otherInput") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input2;
    }
  }
  return mapping;
}

TEST_F(TbeConvDequantS16FusionTest, conv2d_dequant_clipByValue_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data1 = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto dequants16_node = builder.AddNode("AscendDequantS16", "AscendDequantS16",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });


  builder.AddDataEdge(conv_node, 0, dequants16_node, 0);
  builder.AddDataEdge(data1, 0, dequants16_node, 1);
  builder.AddDataEdge(data2, 0, dequants16_node, 2);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvDequantS16(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_dequants16_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);
}