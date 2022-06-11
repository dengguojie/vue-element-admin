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

class Conv2dAddClipMulDivFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_add_clip_mul_div_test SetUp" << std::endl;
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
    std::cout << "conv2d_add_clip_mul_div_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfConv2dAddClipMulDiv(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> Conv2dAddClipMulDivFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dAddClipMulDivFusionTest::ptr_buffer_fusion_pass_func;
const string Conv2dAddClipMulDivFusionTest::name_fusion_pass = "TbeConv2dAddClipMulDivFusionPass";
const fe::BufferFusionPassType Conv2dAddClipMulDivFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dAddClipMulDivFusionTest::ConstructFusionMappingOfConv2dAddClipMulDiv(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_add;
  vector<ge::NodePtr> nodes_clip;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_input3;
  vector<ge::NodePtr> nodes_input4;
  vector<ge::NodePtr> nodes_mul;
  vector<ge::NodePtr> nodes_div;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_add.push_back(ptr_node);
    } else if (desc->GetType() == "Clip") {
      nodes_clip.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    } else if (desc->GetName() == "Data3") {
      nodes_input3.push_back(ptr_node);
    } else if (desc->GetName() == "Data4") {
      nodes_input4.push_back(ptr_node);
    } else if (desc->GetName() == "Mul") {
      nodes_mul.push_back(ptr_node);
    } else if (desc->GetName() == "Div") {
      nodes_div.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "ConvAddClipMulDivBroadcastFusionPass", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "add") {
      mapping[desc] = nodes_add;
    } else if (desc->desc_name == "clip") {
      mapping[desc] = nodes_clip;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput2") {
      mapping[desc] = nodes_input2;
    } else if (desc->desc_name == "otherInput3") {
      mapping[desc] = nodes_input3;
    } else if (desc->desc_name == "otherInput4") {
      mapping[desc] = nodes_input4;
    } else if (desc->desc_name == "mul") {
      mapping[desc] = nodes_mul;
    } else if (desc->desc_name == "div") {
      mapping[desc] = nodes_div;
    }
  }
  return mapping;
}

TEST_F(Conv2dAddClipMulDivFusionTest, conv2d_add_clip_mul_div_fusion_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1}, FORMAT_ND, ge::DT_FLOAT16);
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto data2 = builder.AddNode("Data3", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto data3 = builder.AddNode("Data4", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto add_node = builder.AddNode("Add", "Add",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                    {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto clip_node = builder.AddNode("Clip", "Clip",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   });
  auto mul_node = builder.AddNode("Mul", "Mul",
                                   {
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto div_node = builder.AddNode("Div", "Div",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                    {Format::FORMAT_ND, ge::DT_FLOAT16, {1}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  builder.AddDataEdge(conv_node, 0, add_node, 0);
  builder.AddDataEdge(conv_node, 0, mul_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 1);
  builder.AddDataEdge(data, 0, div_node, 1);
  builder.AddDataEdge(add_node, 0, clip_node, 0);
  builder.AddDataEdge(data2, 0, clip_node, 1);
  builder.AddDataEdge(data3, 0, clip_node, 2);
  builder.AddDataEdge(clip_node, 0, mul_node, 0);
  builder.AddDataEdge(mul_node, 0, div_node, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConv2dAddClipMulDiv(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "conv2d_add_clip_mul_div_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  //EXPECT_EQ(res, fe::SUCCESS);                                                   
}
