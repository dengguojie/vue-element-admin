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

class Conv2dAddReluMulMulFustionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_add_relu6_mul_mul_test SetUp" << std::endl;
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
    std::cout << "conv2d_add_relu6_mul_mul_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfConv2dAddReluMulMul(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> Conv2dAddReluMulMulFustionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> Conv2dAddReluMulMulFustionTest::ptr_buffer_fusion_pass_func;
const string Conv2dAddReluMulMulFustionTest::name_fusion_pass = "TbeConv2dAddRelu6MulMulFusionPass";
const fe::BufferFusionPassType Conv2dAddReluMulMulFustionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping Conv2dAddReluMulMulFustionTest::ConstructFusionMappingOfConv2dAddReluMulMul(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_add;
  vector<ge::NodePtr> nodes_relu;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_input3;
  vector<ge::NodePtr> nodes_mul1;
  vector<ge::NodePtr> nodes_mul2;
  vector<ge::NodePtr> nodes_output;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_add.push_back(ptr_node);
    } else if (desc->GetType() == "Relu6") {
      nodes_relu.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    } else if (desc->GetName() == "Data3") {
      nodes_input3.push_back(ptr_node);
    } else if (desc->GetName() == "Mul1") {
      nodes_mul1.push_back(ptr_node);
    } else if (desc->GetName() == "Mul2") {
      nodes_mul2.push_back(ptr_node);
    } else if (desc->GetName() == "Mul3") {
      nodes_output.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "ConvAddRelu6MulMulBroadcastFusionPass", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "conv2d") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "add") {
      mapping[desc] = nodes_add;
    } else if (desc->desc_name == "relu6") {
      mapping[desc] = nodes_relu;
    } else if (desc->desc_name == "otherInput") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input2;
    } else if (desc->desc_name == "otherInput2") {
      mapping[desc] = nodes_input3;
    } else if (desc->desc_name == "mul1") {
      mapping[desc] = nodes_mul1;
    } else if (desc->desc_name == "mul2") {
      mapping[desc] = nodes_mul2;
    } else if (desc->desc_name == "otherOutput") {
      mapping[desc] = nodes_output;
    }
  }
  return mapping;
}

TEST_F(Conv2dAddReluMulMulFustionTest, conv2d_add_relu6_mul_mul_fusion_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1}, FORMAT_ND, ge::DT_FLOAT16);
  auto data2 = builder.AddNode("Data3", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto data = builder.AddNode("Data1", "Conv2D",
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
                                    {Format::FORMAT_ND, ge::DT_FLOAT16, {1}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto relu_node = builder.AddNode("Relu6", "Relu6",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   });
  auto mul1_node = builder.AddNode("Mul1", "Mul",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul2_node = builder.AddNode("Mul2", "Mul",
                                   {
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul3_node = builder.AddNode("Mul3", "Mul",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  builder.AddDataEdge(conv_node, 0, add_node, 0);
  builder.AddDataEdge(conv_node, 0, mul3_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 1);
  builder.AddDataEdge(data, 0, mul1_node, 0);
  builder.AddDataEdge(data2, 0, mul2_node, 0);
  builder.AddDataEdge(add_node, 0, relu_node, 0);
  builder.AddDataEdge(relu_node, 0, mul1_node, 1);
  builder.AddDataEdge(mul1_node, 0, mul2_node, 1);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConv2dAddReluMulMul(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "conv2d_add_relu6_mul_mul_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);                                                       
}
