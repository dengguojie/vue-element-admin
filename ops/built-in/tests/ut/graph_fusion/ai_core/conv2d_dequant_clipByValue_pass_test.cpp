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

class Conv2dDequantClipByValueFusionTest : public testing::Test {
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
  static fe::BufferFusionMapping ConstructFusionMappingOfConv2dDequantClipByValue(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};

vector<fe::BufferFusionPattern *> Conv2dDequantClipByValueFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dDequantClipByValueFusionTest::ptr_buffer_fusion_pass_func;
const string Conv2dDequantClipByValueFusionTest::name_fusion_pass = "Conv2DDequantClipByValueFusionPass";
const fe::BufferFusionPassType Conv2dDequantClipByValueFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dDequantClipByValueFusionTest::ConstructFusionMappingOfConv2dDequantClipByValue(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_dequant;
  vector<ge::NodePtr> nodes_quant;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_input3;
  vector<ge::NodePtr> nodes_elemwise;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "AscendDequant") {
      nodes_dequant.push_back(ptr_node);
    } else if (desc->GetType() == "Elemwise") {
      nodes_elemwise.push_back(ptr_node);
    } else if (desc->GetType() == "AscendQuant") {
      nodes_quant.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    } else if (desc->GetName() == "Data3") {
      nodes_input3.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "Conv2DDequantClipByValueQuantFusion", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "conv2d") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "dequant") {
      mapping[desc] = nodes_dequant;
    } else if (desc->desc_name == "elemwise") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "quant") {
      mapping[desc] = nodes_quant;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput2") {
      mapping[desc] = nodes_input2;
    } else if (desc->desc_name == "otherInput3") {
      mapping[desc] = nodes_input3;
    }
  }
  return mapping;
}

TEST_F(Conv2dDequantClipByValueFusionTest, conv2d_dequant_clipByValue_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data1 = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data3 = builder.AddNode("Data3", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto dequant_node = builder.AddNode("AscendDequant", "AscendDequant",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto elemwise_node = builder.AddNode("Elemwise", "Elemwise",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto quant_node = builder.AddNode("AscendQuant", "AscendQuant",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });


  builder.AddDataEdge(conv_node, 0, dequant_node, 0);
  builder.AddDataEdge(data1, 0, dequant_node, 1);
  builder.AddDataEdge(dequant_node, 0, elemwise_node, 0);
  builder.AddDataEdge(data2, 0, elemwise_node, 1);
  builder.AddDataEdge(data3, 0, elemwise_node, 2);
  builder.AddDataEdge(elemwise_node, 0, quant_node, 0);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConv2dDequantClipByValue(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "conv2d_dequant_clipByValue_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);
}