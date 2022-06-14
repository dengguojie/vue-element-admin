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

class TbeConvSigmoidMulQuantFusionTest : public testing::Test {
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
  static fe::BufferFusionMapping ConstructFusionMappingOfConvSigmoidMulQunat(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};

vector<fe::BufferFusionPattern *> TbeConvSigmoidMulQuantFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> TbeConvSigmoidMulQuantFusionTest::ptr_buffer_fusion_pass_func;
const string TbeConvSigmoidMulQuantFusionTest::name_fusion_pass = "TbeConvSigmoidMulQuantFusionPass";
const fe::BufferFusionPassType TbeConvSigmoidMulQuantFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping TbeConvSigmoidMulQuantFusionTest::ConstructFusionMappingOfConvSigmoidMulQunat(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_dequant;
  vector<ge::NodePtr> nodes_quant;
  vector<ge::NodePtr> nodes_input;
  vector<ge::NodePtr> nodes_sigmoid;
  vector<ge::NodePtr> nodes_mul;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "AscendDequant") {
      nodes_dequant.push_back(ptr_node);
    } else if (desc->GetName() == "sigmoid") {
      nodes_sigmoid.push_back(ptr_node);
    } else if (desc->GetType() == "AscendQuant") {
      nodes_quant.push_back(ptr_node);
    } else if (desc->GetName() == "Data") {
      nodes_input.push_back(ptr_node);
    } else if (desc->GetName() == "Mul") {
      nodes_mul.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeConvDequantSigmoidMulQuantFusionPass", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "dequant") {
      mapping[desc] = nodes_dequant;
    } else if (desc->desc_name == "sigmoid") {
      mapping[desc] = nodes_sigmoid;
    } else if (desc->desc_name == "quant") {
      mapping[desc] = nodes_quant;
    } else if (desc->desc_name == "otherInput") {
      mapping[desc] = nodes_input;
    } else if (desc->desc_name == "mul") {
      mapping[desc] = nodes_mul;
    }
  }
  return mapping;
}

TEST_F(TbeConvSigmoidMulQuantFusionTest, tbe_conv_sigmoid_mul_quant_test_1) {
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
  auto dequant_node = builder.AddNode("AscendDequant", "AscendDequant",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto sigmoid_node = builder.AddNode("sigmoid", "Sigmoid",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul_node = builder.AddNode("Mul", "Mul",
                                   {
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
  builder.AddDataEdge(data, 0, dequant_node, 1);
  builder.AddDataEdge(dequant_node, 0, sigmoid_node, 0);
  builder.AddDataEdge(sigmoid_node, 0, mul_node, 0);
  builder.AddDataEdge(dequant_node, 0, mul_node, 1);
  builder.AddDataEdge(mul_node, 0, quant_node, 0);


  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfConvSigmoidMulQunat(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_conv_sigmoid_mul_quant_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  EXPECT_EQ(res, fe::SUCCESS);
}