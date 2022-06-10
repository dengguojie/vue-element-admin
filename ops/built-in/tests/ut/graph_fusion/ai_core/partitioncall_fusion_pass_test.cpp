#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class partitioncall_fusion_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "partitioncall_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "partitioncall_fusion_test TearDown" << std::endl;
    }
};

void BuildGraphForPartitionCall(ge::ComputeGraphPtr &parent_graph, ge::ComputeGraphPtr &sub_graph, int32_t caseMode = 0) {

    ge::GeShape data1Shape({1024, 512});
    ge::GeTensorDesc data1Desc(data1Shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    data1Desc.SetOriginFormat(ge::FORMAT_ND);
    data1Desc.SetOriginDataType(ge::DT_FLOAT16);
    data1Desc.SetOriginShape(data1Shape);

    ge::GeShape data2Shape({512, 256});
    ge::GeTensorDesc data2Desc(data2Shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    data2Desc.SetOriginFormat(ge::FORMAT_ND);
    data2Desc.SetOriginDataType(ge::DT_FLOAT16);
    data2Desc.SetOriginShape(data2Shape);

    ge::GeShape callShape({16, 64, 16, 16});
    ge::GeShape callNdShape({1024, 256});
    ge::GeTensorDesc callDesc(callShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    data2Desc.SetOriginFormat(ge::FORMAT_ND);
    data2Desc.SetOriginDataType(ge::DT_FLOAT16);
    data2Desc.SetOriginShape(callNdShape);

    ge::GeShape transShape({1024, 256});
    ge::GeTensorDesc transDesc(transShape, ge::FORMAT_ND, ge::DT_FLOAT16);
    transDesc.SetOriginFormat(ge::FORMAT_ND);
    transDesc.SetOriginDataType(ge::DT_FLOAT16);
    transDesc.SetOriginShape(transShape);
    if (caseMode == 2) {
        transDesc.SetFormat(ge::FORMAT_NCHW);
        transDesc.SetOriginFormat(ge::FORMAT_NCHW);
    }
    ge::OpDescPtr data1 = std::make_shared<ge::OpDesc>("data1", "Data");
    ge::OpDescPtr data2 = std::make_shared<ge::OpDesc>("data2", "Data");
    ge::OpDescPtr func = std::make_shared<ge::OpDesc>("func", "PartitionedCall");
    ge::OpDescPtr transdata = std::make_shared<ge::OpDesc>("transdata", "TransData");
    ge::OpDescPtr output = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data1->AddOutputDesc(data1Desc);
    data2->AddOutputDesc(data2Desc);
    func->AddInputDesc(data1Desc);
    func->AddInputDesc(data2Desc);
    func->AddOutputDesc(callDesc);
    transdata->AddInputDesc(callDesc);
    transdata->AddOutputDesc(transDesc);
    output->AddInputDesc(transDesc);
    output->AddOutputDesc(transDesc);

    parent_graph = std::make_shared<ge::ComputeGraph>("parentgraph");
    ge::NodePtr data1Node = parent_graph->AddNode(data1);
    ge::NodePtr data2Node = parent_graph->AddNode(data2);
    ge::NodePtr funcNode = parent_graph->AddNode(func);
    ge::NodePtr transNode = parent_graph->AddNode(transdata);
    ge::NodePtr output_node = parent_graph->AddNode(output);
    ge::GraphUtils::AddEdge(data1Node->GetOutDataAnchor(0), funcNode->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2Node->GetOutDataAnchor(0), funcNode->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(funcNode->GetOutDataAnchor(0), transNode->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    ge::GeShape subData1Shape({32, 64, 16, 16});
    ge::GeShape subData1OriShape({1024, 512});
    ge::GeTensorDesc subData1Desc(subData1Shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    subData1Desc.SetOriginFormat(ge::FORMAT_ND);
    subData1Desc.SetOriginDataType(ge::DT_FLOAT16);
    subData1Desc.SetOriginShape(subData1OriShape);

    ge::GeShape subData2Shape({16, 32, 16, 16});
    ge::GeShape subData2OriShape({512, 256});
    ge::GeTensorDesc subData2Desc(subData2Shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    subData2Desc.SetOriginFormat(ge::FORMAT_ND);
    subData2Desc.SetOriginDataType(ge::DT_FLOAT16);
    subData2Desc.SetOriginShape(subData2OriShape);

    ge::GeShape matMulShape({16, 64, 16, 16});
    ge::GeShape matMulOriShape({1024, 256});
    ge::GeTensorDesc matMulDesc(matMulShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    matMulDesc.SetOriginFormat(ge::FORMAT_ND);
    matMulDesc.SetOriginDataType(ge::DT_FLOAT16);
    matMulDesc.SetOriginShape(matMulOriShape);

    ge::OpDescPtr subData1 = std::make_shared<ge::OpDesc>("x1", "Data");
    ge::OpDescPtr subData2 = std::make_shared<ge::OpDesc>("x2", "Data");
    ge::OpDescPtr matMulV2 = std::make_shared<ge::OpDesc>("MatMulV2", "MatMulV2");
    if (caseMode == 1) {
        matMulV2 = std::make_shared<ge::OpDesc>("BatchMatMul", "BatchMatMul");
    }
    ge::OpDescPtr subNetoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    subData1->AddOutputDesc(subData1Desc);
    subData2->AddOutputDesc(subData2Desc);
    matMulV2->AddInputDesc("x1", subData1Desc);
    matMulV2->AddInputDesc("x2", subData2Desc);
    matMulV2->AddOutputDesc("y", matMulDesc);
    subNetoutput->AddInputDesc(matMulDesc);
    subNetoutput->AddOutputDesc(matMulDesc);

    sub_graph = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr subData1Node = sub_graph->AddNode(subData1);
    ge::NodePtr subData2Node = sub_graph->AddNode(subData2);
    ge::NodePtr subMatmulNode = sub_graph->AddNode(matMulV2);
    ge::NodePtr subNetoutputNode = sub_graph->AddNode(subNetoutput);

    ge::GraphUtils::AddEdge(subData1Node->GetOutDataAnchor(0), subMatmulNode->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(subData2Node->GetOutDataAnchor(0), subMatmulNode->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(subMatmulNode->GetOutDataAnchor(0), subNetoutputNode->GetInDataAnchor(0));

    funcNode->GetOpDesc()->AddSubgraphName("f");
    funcNode->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    sub_graph->SetParentNode(funcNode);
    sub_graph->SetParentGraph(parent_graph);
    parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
}

TEST_F(partitioncall_fusion_test, partitioncall_fusion_test_1) {
    ge::ComputeGraphPtr parent_graph;
    ge::ComputeGraphPtr sub_graph;
    BuildGraphForPartitionCall(parent_graph, sub_graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("PartitionedCallFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *parent_graph);
    bool findTransData = false;
    for (auto node : parent_graph->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            findTransData = true;
            break;
        }
    }
    EXPECT_EQ(findTransData, true);
}

TEST_F(partitioncall_fusion_test, partitioncall_fusion_test_2) {
    ge::ComputeGraphPtr parent_graph;
    ge::ComputeGraphPtr sub_graph;
    BuildGraphForPartitionCall(parent_graph, sub_graph, 1);
    fe::FusionPassTestUtils::RunGraphFusionPass("PartitionedCallFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *parent_graph);
    bool findTransData = false;
    for (auto node : parent_graph->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            findTransData = true;
            break;
        }
    }
    EXPECT_EQ(findTransData, true);
}

TEST_F(partitioncall_fusion_test, partitioncall_fusion_test_3) {
    ge::ComputeGraphPtr parent_graph;
    ge::ComputeGraphPtr sub_graph;
    BuildGraphForPartitionCall(parent_graph, sub_graph, 2);
    fe::FusionPassTestUtils::RunGraphFusionPass("PartitionedCallFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *parent_graph);
    bool findTransData = false;
    for (auto node : parent_graph->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            findTransData = true;
            break;
        }
    }
    EXPECT_EQ(findTransData, true);
}