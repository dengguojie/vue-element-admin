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

class TbeConvDoubleInFusionTest : public testing::Test {
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
  static fe::BufferFusionMapping ConstructFusionMappingOfConvDoubleIn(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};

vector<fe::BufferFusionPattern *> TbeConvDoubleInFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> TbeConvDoubleInFusionTest::ptr_buffer_fusion_pass_func;
const string TbeConvDoubleInFusionTest::name_fusion_pass = "TbeConvDoubleInFusionPass";
const fe::BufferFusionPassType TbeConvDoubleInFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping TbeConvDoubleInFusionTest::ConstructFusionMappingOfConvDoubleIn(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_input;
  vector<ge::NodePtr> nodes_elemwise1;
  vector<ge::NodePtr> nodes_elemwise2;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetName() == "Eltwise1") {
      nodes_elemwise1.push_back(ptr_node);
    } else if (desc->GetName() == "Eltwise2") {
      nodes_elemwise2.push_back(ptr_node);
    } else if (desc->GetName() == "Data") {
      nodes_input.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvElemwiseReluFusion", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "eltwise1") {
      mapping[desc] = nodes_elemwise1;
    } else if (desc->desc_name == "eltwise2") {
      mapping[desc] = nodes_elemwise2;
    } else if (desc->desc_name == "otherInput") {
      mapping[desc] = nodes_input;
    }
  }
  return mapping;
}

TEST_F(TbeConvDoubleInFusionTest, conv2d_dequant_clipByValue_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto elemwise1_node = builder.AddNode("Eltwise1", "Eltwise",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto elemwise2_node = builder.AddNode("Eltwise2", "Eltwise",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });


  builder.AddDataEdge(conv_node, 0, elemwise1_node, 0);
  builder.AddDataEdge(data, 0, elemwise1_node, 1);
  builder.AddDataEdge(elemwise1_node, 0, elemwise2_node, 0);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvDoubleIn(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_double_in_fusion_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);
}