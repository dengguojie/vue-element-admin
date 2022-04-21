#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class padd_remove_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "concatv2_aligned_fusion_pass_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "concatv2_aligned_fusion_pass_test TearDown" << std::endl; }
};

static bool CheckPadD(ComputeGraphPtr &graph) {
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "PadD") {
      return true;
    }
  }
  return false;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_normal_case_1) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_normal_case_1 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), false);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_normal_case_1 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_normal_case_2) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_normal_case_2 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2Node = builder.AddNode("ConcatV2", "ConcatV2",
                                      {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                       {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                      {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2Node, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2Node, 1);
  builder.AddDataEdge(concatv2Node, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2Node->GetOpDesc(), "N", 2);
  GeTensorDesc concatDimDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  int32_t concatDimData = 1;
  auto concatDimTensor =
      std::make_shared<ge::GeTensor>(concatDimDesc, reinterpret_cast<uint8_t *>(&concatDimData), sizeof(concatDimData));
  OpDescUtils::SetWeights(concatv2Node, {concatDimTensor});
  concatv2Node->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), false);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_normal_case_2 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_3) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_3 startt." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 = builder.AddNode("TransData1", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_3 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_4) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_4 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 =
      builder.AddNode("TransData2", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "ND");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_4 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_5) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_5 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {290, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_5 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_6) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_6 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {-1, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {-1, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_6 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_7) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_7 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {2, 2}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_7 successful." << std::endl;
}

TEST_F(padd_remove_fusion_pass_test, padd_remove_abnormal_case_8) {
  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_8 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 4620}));

  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {8, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(Format::FORMAT_ND);
  dataNode2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({640, 128}));

  auto transData1 =
      builder.AddNode("TransData1", "TransData", {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}},
                      {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}});
  auto padD = builder.AddNode("PadD", "PadD", {{Format::FORMAT_ND, DT_FLOAT, {640, 4620}}},
                              {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}});
  auto transData2 = builder.AddNode("TransData2", "TransData", {{Format::FORMAT_ND, DT_FLOAT, {640, 4624}}},
                                    {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}}});
  auto concatv2DNode = builder.AddNode("ConcatV2D", "ConcatV2D",
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {289, 40, 16, 16}},
                                        {Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {8, 40, 16, 16}}},
                                       {{Format::FORMAT_FRACTAL_NZ, DT_FLOAT, {297, 40, 16, 16}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {297, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT);
  auto reluNode = builder.AddNode("Relu", "Relu", 1, 1, {640, 4624}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, transData1, 0);
  builder.AddDataEdge(transData1, 0, padD, 0);
  builder.AddDataEdge(padD, 0, transData2, 0);
  builder.AddDataEdge(padD, 0, reluNode, 0);
  builder.AddDataEdge(transData2, 0, concatv2DNode, 0);
  builder.AddDataEdge(dataNode2, 0, concatv2DNode, 1);
  builder.AddDataEdge(concatv2DNode, 0, netOutput, 0);

  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transData1->GetOpDesc(), "dst_format", "ND");

  vector<vector<int64_t>> paddingsValue = {{0, 0}, {0, 4}};
  ge::AttrUtils::SetListListInt(padD->GetOpDesc(), "paddings", paddingsValue);

  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "src_format", "ND");
  ge::AttrUtils::SetStr(transData2->GetOpDesc(), "dst_format", "FRACTAL_NZ");

  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "concat_dim", 1);
  ge::AttrUtils::SetInt(concatv2DNode->GetOpDesc(), "N", 2);
  concatv2DNode->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}});

  auto graph = builder.GetGraph();
  fe::FusionPassTestUtils::RunGraphFusionPass("APadDRemoveFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *graph);

  EXPECT_EQ(CheckPadD(graph), true);

  std::cout << "padd_remove_fusion_pass_test.padd_remove_abnormal_case_8 successful." << std::endl;
}