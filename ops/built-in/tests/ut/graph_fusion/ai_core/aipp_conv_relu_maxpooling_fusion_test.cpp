#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_conv_relu_maxpooling_fusion_pass.h"
#undef protected
#undef private

using namespace ge;

class AippConvReluMaxpoolingFusionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Aipp conv relu maxpooling fusion pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Aipp conv relu maxpooling fusion pass TearDown" << std::endl;
  }
};

TEST_F(AippConvReluMaxpoolingFusionTest, AippConvReluMaxpoolingFusionTest_1) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr sqrt_op = std::make_shared<OpDesc>("sqrt", "Sqrt");
  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");

  vector<int64_t> dim({40, 25, 7, 7});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShape(shape);

  relu_op->AddInputDesc("x", tensor_desc);
  relu_op->AddOutputDesc("y", tensor_desc);
  sqrt_op->AddInputDesc("x", tensor_desc);
  sqrt_op->AddOutputDesc("y", tensor_desc);
  relu6_op->AddInputDesc("x", tensor_desc);
  relu6_op->AddOutputDesc("y", tensor_desc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sqrt_node = graph->AddNode(sqrt_op);
  NodePtr relu6_node = graph->AddNode(relu6_op);
  fe::TbeAippConvReluMaxpoolingFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc fusion_desc;
  fusion_desc.desc_name = "elemwise";
  const fe::BufferFusionOpDesc *fusion_desc_ptr = &fusion_desc;
  fe::BufferFusionMapping mapping;
  vector<NodePtr> elemwise_nodes = {relu_node, sqrt_node, relu6_node};
  mapping.emplace(fusion_desc_ptr, elemwise_nodes);
  vector<NodePtr> fusion_nodes;

  fe::Status status = fusion_pass.GetFusionNodes(mapping, fusion_nodes);
  fusion_desc_ptr = nullptr;
}

TEST_F(AippConvReluMaxpoolingFusionTest, AippConvReluMaxpoolingFusionTest_2) {
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

  fe::TbeAippConvReluMaxpoolingFusionPass fusion_pass;
  fusion_pass.CheckConvNodeValidation(conv_node);

}

TEST_F(AippConvReluMaxpoolingFusionTest, AippConvReluMaxpoolingFusionTest_3) {
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

  fe::TbeAippConvReluMaxpoolingFusionPass fusion_pass;
  fusion_pass.CheckMaxpoolNodeValidation(conv_node);

}


TEST_F(AippConvReluMaxpoolingFusionTest, AippConvReluMaxpoolingFusionTest_4) {
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
  NodePtr aipp_node = graph->AddNode(conv_op);
  NodePtr max_pool_node = graph->AddNode(conv_op);

  fe::TbeAippConvReluMaxpoolingFusionPass fusion_pass;
  fusion_pass.PoolingValidationAndFormatSet(aipp_node,conv_node,max_pool_node);
}

TEST_F(AippConvReluMaxpoolingFusionTest, AippConvReluMaxpoolingFusionTest_5) {
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

  fe::TbeAippConvReluMaxpoolingFusionPass fusion_pass;
  fusion_pass.CheckConvNodeValidation(conv_node);

}