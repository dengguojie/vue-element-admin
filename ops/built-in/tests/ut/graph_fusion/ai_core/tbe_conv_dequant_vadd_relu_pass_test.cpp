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

class Conv2dDequantVaddReluFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_dequant_vadd_relu_pass_test SetUp" << std::endl;
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
    std::cout << "conv2d_dequant_vadd_relu_pass_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfConvDequantVaddRelu(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> Conv2dDequantVaddReluFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dDequantVaddReluFusionTest::ptr_buffer_fusion_pass_func;
const string Conv2dDequantVaddReluFusionTest::name_fusion_pass = "TbeConvDequantVaddReluFusionPass";
const fe::BufferFusionPassType Conv2dDequantVaddReluFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dDequantVaddReluFusionTest::ConstructFusionMappingOfConvDequantVaddRelu(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_dequant;
  vector<ge::NodePtr> nodes_vadd;
  vector<ge::NodePtr> nodes_relu;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_output;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "AscendDequant") {
      nodes_dequant.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_vadd.push_back(ptr_node);
    } else if (desc->GetType() == "Relu") {
      nodes_relu.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    } else if (desc->GetName() == "NetOutput") {
      nodes_output.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvDequantVaddReluBroadcastFusion", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "dequant") {
      mapping[desc] = nodes_dequant;
    } else if (desc->desc_name == "vadd") {
      mapping[desc] = nodes_vadd;
    } else if (desc->desc_name == "relu") {
      mapping[desc] = nodes_relu;
    } else if (desc->desc_name == "otherInput") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input2;
    } else if (desc->desc_name == "otherOutput") {
      mapping[desc] = nodes_output;
    }
  }
  return mapping;
}

TEST_F(Conv2dDequantVaddReluFusionTest, conv2d_dequant_vadd_relu_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
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
  auto add_node = builder.AddNode("Add", "Add",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                    {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto relu_node = builder.AddNode("Relu", "Relu",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto net_output = builder.AddNode("NetOutput", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  builder.AddDataEdge(conv_node, 0, dequant_node, 0);
  builder.AddDataEdge(data, 0, dequant_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 1);
  builder.AddDataEdge(dequant_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, relu_node, 0);
  builder.AddDataEdge(relu_node, 0, net_output, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvDequantVaddRelu(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_dequant_vadd_relu_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}

TEST_F(Conv2dDequantVaddReluFusionTest, conv2d_dequant_vadd_relu_pass_test_2) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1}, FORMAT_ND, ge::DT_FLOAT16);
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
  auto add_node = builder.AddNode("Add", "Add",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                    {Format::FORMAT_ND, ge::DT_FLOAT16, {1}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto relu_node = builder.AddNode("Relu", "Relu",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto net_output = builder.AddNode("NetOutput", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  builder.AddDataEdge(conv_node, 0, dequant_node, 0);
  builder.AddDataEdge(data, 0, dequant_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 1);
  builder.AddDataEdge(dequant_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, relu_node, 0);
  builder.AddDataEdge(relu_node, 0, net_output, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvDequantVaddRelu(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_dequant_vadd_relu_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}

TEST_F(Conv2dDequantVaddReluFusionTest, conv2d_dequant_vadd_relu_pass_test_3) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
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
  auto add_node = builder.AddNode("Add", "Add",
                                   {
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto relu_node = builder.AddNode("Relu", "Relu",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto net_output = builder.AddNode("NetOutput", "NetOutput",
                                    1, 0, {1, 1, 56, 56, 16}, FORMAT_NC1HWC0, ge::DT_FLOAT16);
  builder.AddDataEdge(conv_node, 0, dequant_node, 0);
  builder.AddDataEdge(data, 0, dequant_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 0);
  builder.AddDataEdge(dequant_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, relu_node, 0);
  builder.AddDataEdge(relu_node, 0, net_output, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvDequantVaddRelu(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_dequant_vadd_relu_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}
