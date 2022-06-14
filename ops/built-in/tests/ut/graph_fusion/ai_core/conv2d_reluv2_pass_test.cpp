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

class Conv2dRelu2FusionTest : public testing::Test {
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
  static fe::BufferFusionMapping ConstructFusionMappingOfConv2dRelu2(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};

vector<fe::BufferFusionPattern *> Conv2dRelu2FusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dRelu2FusionTest::ptr_buffer_fusion_pass_func;
const string Conv2dRelu2FusionTest::name_fusion_pass = "TbeConv2DReluv2Pass";
const fe::BufferFusionPassType Conv2dRelu2FusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dRelu2FusionTest::ConstructFusionMappingOfConv2dRelu2(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_relu;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "Reluv2") {
      nodes_relu.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConv2DReluv2Fusion", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "conv2d") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "reluv2") {
      mapping[desc] = nodes_relu;
    }
  }
  return mapping;
}

TEST_F(Conv2dRelu2FusionTest, conv2d_relu2_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto relu_node = builder.AddNode("Reluv2", "Reluv2",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });


  builder.AddDataEdge(conv_node, 0, relu_node, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConv2dRelu2(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "conv2d_reluv2_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);
}