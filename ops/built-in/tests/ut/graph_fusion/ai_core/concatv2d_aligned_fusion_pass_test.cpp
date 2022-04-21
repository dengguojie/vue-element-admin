#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class concatv2d_aligned_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "concatv2_aligned_fusion_pass_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "concatv2_aligned_fusion_pass_test TearDown" << std::endl; }
};

static bool CheckPadD1(ComputeGraphPtr &graph) {
  std::vector<int64_t> expectedDims = {640, 4624};
  ge::DataType expectDatatype = DT_FLOAT;

  bool findPadD = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "Data1/PadD") {
      findPadD = true;
      auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedDims);
      EXPECT_EQ(outputDesc.GetDataType(), expectDatatype);
      break;
    }
  }
  return findPadD;
}

static bool CheckPadD2(ComputeGraphPtr &graph) {
  std::vector<int64_t> expectedDims = {4624, 1};
  ge::DataType expectDatatype = DT_FLOAT;

  bool findPadD = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "constant1/SplitV/PadD") {
      findPadD = true;
      auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedDims);
      EXPECT_EQ(outputDesc.GetDataType(), expectDatatype);
      break;
    }
  }
  return findPadD;
}

static bool CheckSplitV(ComputeGraphPtr &graph) {
  std::vector<int64_t> expectedDims1 = {4620, 1};
  std::vector<int64_t> expectedDims2 = {128, 1};
  DataType expectDatatype = DT_FLOAT;

  bool findSplitV = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "constant1/SplitV") {
      findSplitV = true;
      auto outputDesc1 = node->GetOpDesc()->GetOutputDesc(0);
      auto outputDesc2 = node->GetOpDesc()->GetOutputDesc(1);
      EXPECT_EQ(outputDesc1.GetShape().GetDims(), expectedDims1);
      EXPECT_EQ(outputDesc2.GetShape().GetDims(), expectedDims2);
      EXPECT_EQ(outputDesc1.GetDataType(), expectDatatype);
      break;
    }
  }
  return findSplitV;
}

static bool CheckConcatV2(ComputeGraphPtr &graph) {
  std::vector<int64_t> expectedDims = {4752, 1};
  ge::DataType expectDatatype = DT_FLOAT;

  bool findConcatV2 = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "constant1/SplitV/PadD/ConcatV2") {
      findConcatV2 = true;
      auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedDims);
      EXPECT_EQ(outputDesc.GetDataType(), expectDatatype);
      break;
    }
  }
  return findConcatV2;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_normal_case_1) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_normal_case_1 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatV2Node = builder.AddNode(
      "ConcatV2", "ConcatV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}, {Format::FORMAT_ND, DT_FLOAT, {640, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4748}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}, {Format::FORMAT_ND, DT_FLOAT, {4748, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatV2Node, 0);
  builder.AddDataEdge(dataNode2, 0, concatV2Node, 1);
  builder.AddDataEdge(concatV2Node, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatV2Node->GetOpDesc(), "N", 2);
  GeTensorDesc concatDimDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  int32_t concatDimData = 1;
  auto concatDimTensor =
      std::make_shared<ge::GeTensor>(concatDimDesc, reinterpret_cast<uint8_t *>(&concatDimData), sizeof(concatDimData));
  OpDescUtils::SetWeights(concatV2Node, {concatDimTensor});
  concatV2Node->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), true);
  EXPECT_EQ(CheckPadD2(graph), true);
  EXPECT_EQ(CheckSplitV(graph), true);
  EXPECT_EQ(CheckConcatV2(graph), true);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_normal_case_1 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_normal_case_2) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_normal_case_2 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatv2DNode = builder.AddNode(
      "ConcatV2D", "ConcatV2D", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}, {Format::FORMAT_ND, DT_FLOAT, {640, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4748}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}, {Format::FORMAT_ND, DT_FLOAT, {4748, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), true);
  EXPECT_EQ(CheckPadD2(graph), true);
  EXPECT_EQ(CheckSplitV(graph), true);
  EXPECT_EQ(CheckConcatV2(graph), true);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_normal_case_2 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_abnormal_case_3) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_3 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 130}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatv2DNode = builder.AddNode(
      "ConcatV2D", "ConcatV2D", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}, {Format::FORMAT_ND, DT_FLOAT, {640, 130}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 4750}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4750}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4750}}, {Format::FORMAT_ND, DT_FLOAT, {4750, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), false);
  EXPECT_EQ(CheckPadD2(graph), false);
  EXPECT_EQ(CheckSplitV(graph), false);
  EXPECT_EQ(CheckConcatV2(graph), false);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_3 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_abnormal_case_4) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_4 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4624}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4752, 1}, FORMAT_ND, DT_FLOAT);

  auto concatv2DNode = builder.AddNode(
      "ConcatV2D", "ConcatV2D", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}, {Format::FORMAT_ND, DT_FLOAT, {640, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 4752}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4752}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4752}}, {Format::FORMAT_ND, DT_FLOAT, {4752, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), false);
  EXPECT_EQ(CheckPadD2(graph), false);
  EXPECT_EQ(CheckSplitV(graph), false);
  EXPECT_EQ(CheckConcatV2(graph), false);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_4 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_abnormal_case_5) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_5 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {1, 640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {1, 640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatV2Node =
      builder.AddNode("ConcatV2", "ConcatV2",
                      {{Format::FORMAT_ND, DT_FLOAT, {1, 640, 4620}}, {Format::FORMAT_ND, DT_FLOAT, {1, 640, 128}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {1, 640, 4748}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {1, 640, 4748}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}, {Format::FORMAT_ND, DT_FLOAT, {4748, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatV2Node, 0);
  builder.AddDataEdge(dataNode2, 0, concatV2Node, 1);
  builder.AddDataEdge(concatV2Node, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatV2Node->GetOpDesc(), "N", 2);
  GeTensorDesc concatDimDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  int32_t concatDimData = 1;
  auto concatDimTensor =
      std::make_shared<ge::GeTensor>(concatDimDesc, reinterpret_cast<uint8_t *>(&concatDimData), sizeof(concatDimData));
  OpDescUtils::SetWeights(concatV2Node, {concatDimTensor});
  concatV2Node->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), false);
  EXPECT_EQ(CheckPadD2(graph), false);
  EXPECT_EQ(CheckSplitV(graph), false);
  EXPECT_EQ(CheckConcatV2(graph), false);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_normal_case_1 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_abnormal_case_6) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_6 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto dataNode3 = builder.AddNode("Data3", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4876, 1}, FORMAT_ND, DT_FLOAT);

  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}},
                                        {Format::FORMAT_ND, DT_FLOAT, {640, 128}},
                                        {Format::FORMAT_ND, DT_FLOAT, {640, 128}}},
                                       {{Format::FORMAT_ND, DT_FLOAT, {640, 4876}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4876}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4876}}, {Format::FORMAT_ND, DT_FLOAT, {4876, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 3);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"x2", 2}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), false);
  EXPECT_EQ(CheckPadD2(graph), false);
  EXPECT_EQ(CheckSplitV(graph), false);
  EXPECT_EQ(CheckConcatV2(graph), false);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_6 successful." << std::endl;
}

TEST_F(concatv2d_aligned_fusion_pass_test, concatv2d_aligned_abnormal_case_7) {
  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_7 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {640, 4620}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {640, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatV2Node = builder.AddNode(
      "ConcatV2", "ConcatV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}, {Format::FORMAT_ND, DT_FLOAT, {640, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}});
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {-1, 4748}, FORMAT_ND, DT_FLOAT);
  auto matmulv2Node = builder.AddNode(
      "MatMulV2", "MatMulV2", {{Format::FORMAT_ND, DT_FLOAT, {640, 4748}}, {Format::FORMAT_ND, DT_FLOAT, {4748, 1}}},
      {{Format::FORMAT_ND, DT_FLOAT, {640, 1}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {640, 1}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatV2Node, 0);
  builder.AddDataEdge(dataNode2, 0, concatV2Node, 1);
  builder.AddDataEdge(concatV2Node, 0, reluNode, 0);
  builder.AddDataEdge(reluNode, 0, matmulv2Node, 0);
  builder.AddDataEdge(constantNode1, 0, matmulv2Node, 1);
  builder.AddDataEdge(matmulv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatV2Node->GetOpDesc(), "N", 2);
  GeTensorDesc concatDimDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  int32_t concatDimData = 1;
  auto concatDimTensor =
      std::make_shared<ge::GeTensor>(concatDimDesc, reinterpret_cast<uint8_t *>(&concatDimData), sizeof(concatDimData));
  OpDescUtils::SetWeights(concatV2Node, {concatDimTensor});
  concatV2Node->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulv2Node->GetOpDesc(), "transpose_x2", false);

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatV2DAlignedFusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD1(graph), false);
  EXPECT_EQ(CheckPadD2(graph), false);
  EXPECT_EQ(CheckSplitV(graph), false);
  EXPECT_EQ(CheckConcatV2(graph), false);

  std::cout << "concatv2d_aligned_fusion_pass_test.concatv2d_aligned_abnormal_case_7 successful." << std::endl;
}
