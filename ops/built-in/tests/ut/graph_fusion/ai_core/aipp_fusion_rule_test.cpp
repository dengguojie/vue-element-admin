#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_fusion_rule.h"
#undef protected
#undef private

using namespace ge;

class AippFusionRuleTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Aipp fusion rule SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Aipp fusion rule TearDown" << std::endl;
  }
};

TEST_F(AippFusionRuleTest, AippFusionRuleTest_1) {
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr transdata_op = std::make_shared<OpDesc>("transdata", "TransData");
  
  vector<int64_t> dim1({1, 1, 15, 33, 4});
  GeShape shape1(dim1);
  vector<int64_t> dim1_ori({1, 3, 15, 33});
  GeShape shape1_ori(dim1_ori);
  GeTensorDesc tensor_desc_1(shape1);
  tensor_desc_1.SetOriginFormat(FORMAT_NC1HWC0_C04);
  tensor_desc_1.SetOriginDataType(DT_FLOAT16);
  tensor_desc_1.SetOriginShape(shape1);
  tensor_desc_1.SetFormat(FORMAT_NCHW);
  tensor_desc_1.SetDataType(DT_FLOAT16);
  tensor_desc_1.SetShape(shape1_ori);

  vector<int64_t> dim2({9, 1, 16, 4});
  GeShape shape2(dim2);
  vector<int64_t> dim2_ori({1, 3, 3, 3});
  GeShape shape2_ori(dim2_ori);
  GeTensorDesc tensor_desc_2(shape2);
  tensor_desc_2.SetOriginFormat(FORMAT_FRACTAL_Z_C04);
  tensor_desc_2.SetOriginDataType(DT_FLOAT16);
  tensor_desc_2.SetOriginShape(shape2);
  tensor_desc_2.SetFormat(FORMAT_NCHW);
  tensor_desc_2.SetDataType(DT_FLOAT16);
  tensor_desc_2.SetShape(shape2_ori);

  conv_op->AddInputDesc("x", tensor_desc_1);
  conv_op->AddInputDesc("filter", tensor_desc_2);
  ge::AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr conv_node = graph->AddNode(conv_op);

  fe::TbeAippFusionRule fusion_rule;
  fusion_rule.CheckAippConvStridehValidation(conv_node);
  fusion_rule.CheckConvload2dNodeValidation(conv_node);
  fusion_rule.CheckAippConvEltwiseFusionValidation(conv_node,"NCHW");

}

TEST_F(AippFusionRuleTest, AippFusionRuleTest_2) {
  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr transdata_op = std::make_shared<OpDesc>("transdata", "TransData");
  
  vector<int64_t> dim1({1, 3, 224, 224});
  GeShape shape1(dim1);
  vector<int64_t> dim1_ori({1, 3, 15, 33});
  GeShape shape1_ori(dim1_ori);
  GeTensorDesc tensor_desc_1(shape1);
  tensor_desc_1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_1.SetOriginDataType(DT_FLOAT16);
  tensor_desc_1.SetOriginShape(shape1);
  tensor_desc_1.SetFormat(FORMAT_NCHW);
  tensor_desc_1.SetDataType(DT_FLOAT16);
  tensor_desc_1.SetShape(shape1_ori);

  vector<int64_t> dim2({9, 1, 16, 4});
  GeShape shape2(dim2);
  vector<int64_t> dim2_ori({1, 3, 3, 3});
  GeShape shape2_ori(dim2_ori);
  GeTensorDesc tensor_desc_2(shape2);
  tensor_desc_2.SetOriginFormat(FORMAT_FRACTAL_Z_C04);
  tensor_desc_2.SetOriginDataType(DT_FLOAT16);
  tensor_desc_2.SetOriginShape(shape2);
  tensor_desc_2.SetFormat(FORMAT_NCHW);
  tensor_desc_2.SetDataType(DT_FLOAT16);
  tensor_desc_2.SetShape(shape2_ori);

  conv_op->AddInputDesc("x", tensor_desc_1);
  conv_op->AddInputDesc("filter", tensor_desc_2);
  ge::AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr conv_node = graph->AddNode(conv_op);

  fe::TbeAippFusionRule fusion_rule;
  fusion_rule.CheckAippConvStridehValidation(conv_node);
  fusion_rule.CheckConvload2dNodeValidation(conv_node);
  fusion_rule.CalcMinAIPPTbeL1Space(conv_node);
  fusion_rule.CheckAippConvEltwiseFusionValidation(conv_node,"NCHW");

}
