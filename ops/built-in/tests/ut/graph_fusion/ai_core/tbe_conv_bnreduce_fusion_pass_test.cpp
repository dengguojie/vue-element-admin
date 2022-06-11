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

class Conv2dBnreduceFussionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "tbe_conv2d_bnreduce_fusion_pass_test SetUp" << std::endl;
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
    std::cout << "tbe_conv2d_bnreduce_fusion_pass_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfConv2dBnreduce(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> Conv2dBnreduceFussionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dBnreduceFussionTest::ptr_buffer_fusion_pass_func;
const string Conv2dBnreduceFussionTest::name_fusion_pass = "TbeConvBnreduceFusionPass";
const fe::BufferFusionPassType Conv2dBnreduceFussionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dBnreduceFussionTest::ConstructFusionMappingOfConv2dBnreduce(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_bnreduce;
  vector<ge::NodePtr> nodes_output1;
  vector<ge::NodePtr> nodes_output2;
  vector<ge::NodePtr> nodes_output3;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "BNTrainingReduce") {
      nodes_bnreduce.push_back(ptr_node);
    } else if (desc->GetName() == "NetOutput1") {
      nodes_output1.push_back(ptr_node);
    } else if (desc->GetName() == "NetOutput2") {
      nodes_output2.push_back(ptr_node);
    } else if (desc->GetName() == "NetOutput3") {
      nodes_output3.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvBNReduceTuplePatternFusionPass", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "bnreduce") {
      mapping[desc] = nodes_bnreduce;
    } else if (desc->desc_name == "OUTPUT1") {
      mapping[desc] = nodes_output1;
    } else if (desc->desc_name == "OUTPUT2") {
      mapping[desc] = nodes_output2;
    } else if (desc->desc_name == "OUTPUT3") {
      mapping[desc] = nodes_output3;
    }
  }
  return mapping;
}

TEST_F(Conv2dBnreduceFussionTest, tbe_conv2d_bnreduce_fusion_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto bnreduce_node = builder.AddNode("BNTrainingReduce", "BNTrainingReduce",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT, {1, 1, 56, 56, 16}},
                                   });

  auto net_output1 = builder.AddNode("NetOutput1", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto net_output2 = builder.AddNode("NetOutput2", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto net_output3 = builder.AddNode("NetOutput3", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  builder.AddDataEdge(conv_node, 0, bnreduce_node, 0);
  builder.AddDataEdge(conv_node, 0, net_output1, 0);
  builder.AddDataEdge(conv_node, 0, net_output2, 0);
  builder.AddDataEdge(conv_node, 0, net_output3, 0);


  auto graph = builder.GetGraph();
  auto mapping = ConstructFusionMappingOfConv2dBnreduce(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv2d_bnreduce_fusion_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}