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


class TbeAippConv2dAddRelu6MulMulFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "tbe_aipp_conv2d_add_relu6_mul_mul_test SetUp" << std::endl;
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
    std::cout << "tbe_aipp_conv2d_add_relu6_mul_mul_test TearDown" << std::endl;
  }
  static fe::BufferFusionMapping ConstructFusionMappingOfAippConv2dAddReluMulMul(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
private:


};

vector<fe::BufferFusionPattern *> TbeAippConv2dAddRelu6MulMulFusionTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> TbeAippConv2dAddRelu6MulMulFusionTest::ptr_buffer_fusion_pass_func;
const string TbeAippConv2dAddRelu6MulMulFusionTest::name_fusion_pass = "TbeAippConv2dAddRelu6MulMulFusionPass";
const fe::BufferFusionPassType TbeAippConv2dAddRelu6MulMulFusionTest::type_fusion_pass = fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping TbeAippConv2dAddRelu6MulMulFusionTest::ConstructFusionMappingOfAippConv2dAddReluMulMul(const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_aipp;
  vector<ge::NodePtr> nodes_conv;
  vector<ge::NodePtr> nodes_add;
  vector<ge::NodePtr> nodes_relu;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_input2;
  vector<ge::NodePtr> nodes_mul1;
  vector<ge::NodePtr> nodes_mul2;
  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetName() == "Aipp") {
      nodes_aipp.push_back(ptr_node);
    } else if (desc->GetType() == "Conv2D") {
      nodes_conv.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_add.push_back(ptr_node);
    } else if (desc->GetType() == "Relu6") {
      nodes_relu.push_back(ptr_node);
    } else if (desc->GetName() == "Data1") {
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "Data2") {
      nodes_input2.push_back(ptr_node);
    } else if (desc->GetName() == "Mul1") {
      nodes_mul1.push_back(ptr_node);
    } else if (desc->GetName() == "Mul2") {
      nodes_mul2.push_back(ptr_node);
    }
  }

  fe::BufferFusionPattern *pattern;
  EXPECT_TRUE(fe::FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeAippConv2dAddRelu6MulMulBroadcastFusionPattern2", &pattern));

  fe::BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "aipp") {
      mapping[desc] = nodes_aipp;
    } else if (desc->desc_name == "convolution") {
      mapping[desc] = nodes_conv;
    } else if (desc->desc_name == "add") {
      mapping[desc] = nodes_add;
    } else if (desc->desc_name == "relu6") {
      mapping[desc] = nodes_relu;
    } else if (desc->desc_name == "otherInput1") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "otherInput2") {
      mapping[desc] = nodes_input2;
    } else if (desc->desc_name == "mul1") {
      mapping[desc] = nodes_mul1;
    } else if (desc->desc_name == "mul2") {
      mapping[desc] = nodes_mul2;
    }
  }
  return mapping;
}

TEST_F(TbeAippConv2dAddRelu6MulMulFusionTest, tbe_aipp_conv2d_add_relu6_mul_mul_fusion_pass_test_1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  auto data = builder.AddNode("Data1", "Data", 0, 1, {1}, FORMAT_ND, ge::DT_FLOAT16);
  auto data1 = builder.AddNode("Data2", "Data", 0, 1, {1, 1, 1, 1, 16}, FORMAT_ND, ge::DT_FLOAT16);
  auto conv_node = builder.AddNode("Conv2D", "Conv2D",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, {9, 1, 16, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto aipp_node = builder.AddNode("Aipp", "Aipp",
                                   {
                                     {Format::FORMAT_NCHW, ge::DT_FLOAT16, {1, 56, 56, 16}},
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
  auto relu_node = builder.AddNode("Relu6", "Relu6",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                   });
  auto mul1_node = builder.AddNode("Mul1", "Mul",
                                   {
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1, 1, 1, 1, 16}},
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  auto mul2_node = builder.AddNode("Mul2", "Mul",
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                     {Format::FORMAT_ND, ge::DT_FLOAT16, {1}},
                                   },
                                   {
                                     {Format::FORMAT_NC1HWC0, ge::DT_FLOAT16, {1, 1, 56, 56, 16}},
                                   });
  builder.AddDataEdge(aipp_node, 0, conv_node, 0);
  builder.AddDataEdge(conv_node, 0, add_node, 0);
  builder.AddDataEdge(conv_node, 0, mul1_node, 1);
  builder.AddDataEdge(data1, 0, add_node, 1);
  builder.AddDataEdge(data, 0, mul2_node, 1);
  builder.AddDataEdge(add_node, 0, relu_node, 0);
  builder.AddDataEdge(relu_node, 0, mul1_node, 0);
  builder.AddDataEdge(mul1_node, 0, mul2_node, 0);

  auto graph = builder.GetGraph();
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfAippConv2dAddReluMulMul(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_aipp_conv2d_add_relu6_mul_mul_pass_before");
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            graph, mapping);
  //EXPECT_EQ(res, fe::SUCCESS);                                                       
}

TEST_F(TbeAippConv2dAddRelu6MulMulFusionTest, tbe_aipp_conv2d_add_relu6_mul_mul_fusion_pass_test_2) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>(this->test_info_->name());

  ge::GeTensorDesc data_out(ge::GeShape({1}), Format::FORMAT_ND, ge::DT_FLOAT16);
  ge::GeTensorDesc aipp_op_out(ge::GeShape({1, 1, 56, 56, 16}), Format::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::GeTensorDesc conv_op_out(ge::GeShape({1, 1, 56, 56, 16}), Format::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::GeTensorDesc add_op_in1(ge::GeShape({1, 1, 56, 56, 16}), Format::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::GeTensorDesc add_op_in2(ge::GeShape({1, 1, 1, 1, 16}), Format::FORMAT_ND, ge::DT_FLOAT16);
  ge::GeTensorDesc add_op_out(ge::GeShape({1, 1, 56, 56, 16}), Format::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::GeTensorDesc relu_op_out(ge::GeShape({1, 1, 1, 1, 16}), Format::FORMAT_ND, ge::DT_FLOAT16);

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("Data1", "Data");
  ge::OpDescPtr data1_op = std::make_shared<ge::OpDesc>("Data2", "Data");
  ge::OpDescPtr conv_op = std::make_shared<ge::OpDesc>("Conv2D", "Conv2D");
  ge::OpDescPtr aipp_op = std::make_shared<ge::OpDesc>("Aipp", "Aipp");
  ge::OpDescPtr add_op = std::make_shared<ge::OpDesc>("Add", "Add");
  ge::OpDescPtr relu_op = std::make_shared<ge::OpDesc>("Relu6", "Relu6");
  ge::OpDescPtr mul1_op = std::make_shared<ge::OpDesc>("Mul1", "Mul");
  ge::OpDescPtr mul2_op = std::make_shared<ge::OpDesc>("Mul2", "Mul");

  aipp_op->AddOutputDesc(aipp_op_out);
  conv_op->AddInputDesc(aipp_op_out);
  conv_op->AddOutputDesc(conv_op_out);
  add_op->AddInputDesc(add_op_in1);
  add_op->AddInputDesc(add_op_in2);
  add_op->AddOutputDesc(add_op_out);
  relu_op->AddInputDesc(add_op_out);
  relu_op->AddOutputDesc(relu_op_out);
  mul1_op->AddInputDesc(conv_op_out);
  mul1_op->AddInputDesc(relu_op_out);
  mul1_op->AddOutputDesc(add_op_in1);
  mul2_op->AddInputDesc(add_op_in1);
  mul2_op->AddInputDesc(data_out);
  data_op->AddOutputDesc(data_out);
  data1_op->AddOutputDesc(add_op_in2);

  //ge::AttrUtils::SetStr(aipp_op, "aipp_config_path", "ops/built-in/tests/common/ci/aipp_config.json");

  ge::NodePtr data = compute_graph_ptr->AddNode(data_op);
  ge::NodePtr data1 = compute_graph_ptr->AddNode(data1_op);
  ge::NodePtr conv_node = compute_graph_ptr->AddNode(conv_op);
  ge::NodePtr aipp_node = compute_graph_ptr->AddNode(aipp_op);
  ge::NodePtr add_node = compute_graph_ptr->AddNode(add_op);
  ge::NodePtr relu_node = compute_graph_ptr->AddNode(relu_op);
  ge::NodePtr mul1_node = compute_graph_ptr->AddNode(mul1_op);
  ge::NodePtr mul2_node = compute_graph_ptr->AddNode(mul2_op);

  ge::GraphUtils::AddEdge(aipp_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
  

  compute_graph_ptr->TopologicalSorting();
  auto graph = compute_graph_ptr;
  // ge::ComputeGraphPtr graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto mapping = ConstructFusionMappingOfAippConv2dAddReluMulMul(graph);
  GraphUtils::DumpGEGraphToOnnx(*graph, "tbe_aipp_conv2d_add_relu6_mul_mul_pass2_before");
  //Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            //graph, mapping);
  //EXPECT_EQ(res, fe::SUCCESS);                                                       
}