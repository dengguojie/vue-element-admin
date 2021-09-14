#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_transdata_pass.h"
#undef protected
#undef private

using namespace ge;

class ConvTransdataFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Conv transdata fusion pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv transdata fusion pass TearDown" << std::endl;
  }
};

TEST_F(ConvTransdataFusionTest, ConvTransdataFusionTest_1) {
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr transdata_op = std::make_shared<OpDesc>("transdata", "TransData");
  
  vector<int64_t> dim1({1, 1, 15, 33, 16});
  GeShape shape1(dim1);
  vector<int64_t> dim1_ori({1, 16, 15, 33});
  GeShape shape1_ori(dim1_ori);
  GeTensorDesc tensor_desc_1(shape1);
  tensor_desc_1.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_desc_1.SetOriginDataType(DT_FLOAT16);
  tensor_desc_1.SetOriginShape(shape1);
  tensor_desc_1.SetFormat(FORMAT_NCHW);
  tensor_desc_1.SetDataType(DT_FLOAT16);
  tensor_desc_1.SetShape(shape1_ori);

  vector<int64_t> dim2({9, 1, 16, 16});
  GeShape shape2(dim2);
  vector<int64_t> dim2_ori({1, 16, 3, 3});
  GeShape shape2_ori(dim2_ori);
  GeTensorDesc tensor_desc_2(shape2);
  tensor_desc_2.SetOriginFormat(FORMAT_FRACTAL_Z);
  tensor_desc_2.SetOriginDataType(DT_FLOAT16);
  tensor_desc_2.SetOriginShape(shape2);
  tensor_desc_2.SetFormat(FORMAT_NCHW);
  tensor_desc_2.SetDataType(DT_FLOAT16);
  tensor_desc_2.SetShape(shape2_ori);

  vector<int64_t> dim3({64});
  GeShape shape3(dim3);
  GeTensorDesc tensor_desc_3(shape3);
  tensor_desc_3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_3.SetOriginDataType(DT_FLOAT16);
  tensor_desc_3.SetOriginShape(shape3);
  tensor_desc_3.SetFormat(FORMAT_NCHW);
  tensor_desc_3.SetDataType(DT_FLOAT16);
  tensor_desc_3.SetShape(shape3);

  vector<int64_t> dim4({1, 1, 7, 16, 16});
  GeShape shape4(dim4);
  vector<int64_t> dim4_ori({1, 1, 7, 16});
  GeShape shape4_ori(dim2_ori);
  GeTensorDesc tensor_desc_4(shape2);
  tensor_desc_4.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_desc_4.SetOriginDataType(DT_FLOAT16);
  tensor_desc_4.SetOriginShape(shape4);
  tensor_desc_4.SetFormat(FORMAT_NCHW);
  tensor_desc_4.SetDataType(DT_FLOAT16);
  tensor_desc_4.SetShape(shape4_ori);

  conv_op->AddInputDesc("x", tensor_desc_1);
  conv_op->AddInputDesc("filter", tensor_desc_2);
  conv_op->AddInputDesc("bias", tensor_desc_3);
  conv_op->AddOutputDesc(tensor_desc_4);
  ge::AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(conv_op, "strides", {1, 1, 2, 2});

  transdata_op->AddInputDesc("src", tensor_desc_1);
  transdata_op->AddOutputDesc("dst", tensor_desc_1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr transdata_node = graph->AddNode(transdata_op);
  ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));

  fe::ConvTransdataFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::FusionPassTestUtils::RunGraphFusionPass("TbeConvTransdataFusion", fe::BUILT_IN_GRAPH_PASS, *graph);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    std::cout << "type:" << node->GetType() << ", name:" << node->GetName() << std::endl;
  }
}
