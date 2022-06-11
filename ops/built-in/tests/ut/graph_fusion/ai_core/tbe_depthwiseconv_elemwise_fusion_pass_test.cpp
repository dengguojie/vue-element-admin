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

class DepthwiseconvElemwiseFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "tbe_depthwiseconv_elemwise_fusion_pass_test SetUp" << std::endl;
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
    std::cout << "tbe_depthwiseconv_elemwise_fusion_pass_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfDepthwiseconvElemwise(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> DepthwiseconvElemwiseFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> DepthwiseconvElemwiseFusionTest::ptr_buffer_fusion_pass_func;
const string DepthwiseconvElemwiseFusionTest::name_fusion_pass = "TbeDepthwiseConvElemwiseFusionPass";
const fe::BufferFusionPassType DepthwiseconvElemwiseFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping DepthwiseconvElemwiseFusionTest::ConstructFusionMappingOfDepthwiseconvElemwise(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_mul;
  vector<ge::NodePtr> nodes_input1;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "DepthwiseConv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "Mul") {
      nodes_mul.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeDepthwiseConvElewiseInputBroadcastFusionPass", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "DepthwiseConvolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "eltwise2") {
      mapping[desc] = nodes_mul;
    } else if (desc->desc_name == "otherInput2") {
      mapping[desc] = nodes_input1;
    }
  }
  return mapping;
}

TEST_F(DepthwiseconvElemwiseFusionTest, depthwiseconv_elemwise_fusion_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "DepthwiseConv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul_node = builder.AddNode("Mul", "Mul",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });

  builder.AddDataEdge(conv_node, 0, mul_node, 0);
  builder.AddDataEdge(data, 0, mul_node, 1);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfDepthwiseconvElemwise(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "depthwiseconv_elemwise_fusion_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}

TEST_F(DepthwiseconvElemwiseFusionTest, depthwiseconv_elemwise_fusion_pass_test_2) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "DepthwiseConv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul_node = builder.AddNode("Mul", "Mul",
                                   {
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });

  builder.AddDataEdge(conv_node, 0, mul_node, 1);
  builder.AddDataEdge(data, 0, mul_node, 0);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfDepthwiseconvElemwise(graph);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}

TEST_F(DepthwiseconvElemwiseFusionTest, depthwiseconv_elemwise_fusion_pass_test_3) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "DepthwiseConv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul_node = builder.AddNode("Mul", "Mul",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });

  builder.AddDataEdge(conv_node, 0, mul_node, 0);
  builder.AddDataEdge(data, 0, mul_node, 1);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfDepthwiseconvElemwise(graph);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}