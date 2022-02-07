#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "reduce_ops.h"
#include "./graph_builder_utils.h"

using namespace ge;

namespace fe {

class square_sum_v2_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "square_sum_v2_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "square_sum_v2_fusion_test TearDown" << std::endl;
  }
};
// error test:square_node->GetOutAllNodes().size()==1
TEST_F(square_sum_v2_fusion_test, square_sum_v2_fusion_test_1) {
  ge::Graph graph("square_sum_v2_fusion_test");
  auto input0 = op::Data("input0");
  std::vector<int64_t> shape{36, 36};
  ge::Shape input0_shape(shape);
  ge::TensorDesc tensorDescinput0(input0_shape, FORMAT_NHWC, DT_FLOAT16);
  input0.update_input_desc_x(tensorDescinput0);
  input0.update_output_desc_y(tensorDescinput0);
  auto square_layer = op::Square("square");
  square_layer.set_input_x(input0);
  auto reduceSumD_layer = op::ReduceSumD("reduceSumD");
  reduceSumD_layer.set_input_x(square_layer).set_attr_axes({1, 1}).set_attr_keep_dims(false);
  std::vector<Operator> inputs{input0};
  std::vector<Operator> outputs{square_layer, reduceSumD_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SquareSumV2", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool square_sum_v2_fusion = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "SquareSumV2") {
      square_sum_v2_fusion = true;
    }
  }
  EXPECT_EQ(square_sum_v2_fusion, false);
}
TEST_F(square_sum_v2_fusion_test, square_sum_v2_fusion_test_2) {
  OpDescPtr input0 = std::make_shared<OpDesc>("input0", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({32, 32}));
    tensorDesc.SetOriginFormat(FORMAT_NHWC);
    input0->AddInputDesc(tensorDesc);
    input0->AddOutputDesc(tensorDesc);
  }

  OpDescPtr output0 = std::make_shared<OpDesc>("output0", "Data");
  {
    GeTensorDesc tensorDesc2(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    tensorDesc2.SetOriginShape(GeShape({32, 32}));
    tensorDesc2.SetOriginFormat(FORMAT_NHWC);
    output0->AddInputDesc(tensorDesc2);
    output0->AddOutputDesc(tensorDesc2);
  }

  OpDescPtr squareOpDesc = std::make_shared<OpDesc>("squareop", "Square");
  {
    GeTensorDesc inputTensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({32, 32}));
    inputTensorDesc.SetOriginFormat(FORMAT_NHWC);

    GeTensorDesc outputTensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({32, 32}));
    outputTensorDesc.SetOriginFormat(FORMAT_NHWC);

    squareOpDesc->AddInputDesc(inputTensorDesc);
    squareOpDesc->AddOutputDesc("x", outputTensorDesc);
  }
  OpDescPtr reduceSumDopDesc = std::make_shared<OpDesc>("reduceSumDop", "ReduceSumD");
  {
    GeTensorDesc inputTensorDesc(GeShape({64, 32}), FORMAT_NHWC, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({64, 32}));
    inputTensorDesc.SetOriginFormat(FORMAT_NHWC);

    GeTensorDesc outputTensorDesc(GeShape({64, 32}), FORMAT_NHWC, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({64, 32}));
    outputTensorDesc.SetOriginFormat(FORMAT_NHWC);

    reduceSumDopDesc->AddInputDesc("y", inputTensorDesc);
    reduceSumDopDesc->AddOutputDesc(outputTensorDesc);
    ge::AttrUtils::SetListInt(reduceSumDopDesc, "axes", {-1, -1});
    ge::AttrUtils::SetBool(reduceSumDopDesc, "keep_dims", false);
  }
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("square_sum_v2_fusion_test");
  NodePtr inputdata_node = graph->AddNode(input0);
  NodePtr outputdata_node = graph->AddNode(output0);
  NodePtr square_node = graph->AddNode(squareOpDesc);
  NodePtr reducesumd_node = graph->AddNode(reduceSumDopDesc);
  GraphUtils::AddEdge(inputdata_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));

  GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), outputdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), reducesumd_node->GetInDataAnchor(0));
  fe::FusionPassTestUtils::RunGraphFusionPass("SquareSumV2", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool square_sum_v2_fusion = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "SquareSumV2") {
      square_sum_v2_fusion = true;
    }
  }
  EXPECT_EQ(square_sum_v2_fusion, true);
}
// error test:sum node is second output node of square
TEST_F(square_sum_v2_fusion_test, square_sum_v2_fusion_test_3) {
  OpDescPtr input0 = std::make_shared<OpDesc>("input0", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({32, 32}));
    tensorDesc.SetOriginFormat(FORMAT_NHWC);
    input0->AddInputDesc(tensorDesc);
    input0->AddOutputDesc(tensorDesc);
  }
  OpDescPtr output0 = std::make_shared<OpDesc>("output0", "Data");
  {
    GeTensorDesc tensorDesc2(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    tensorDesc2.SetOriginShape(GeShape({32, 32}));
    tensorDesc2.SetOriginFormat(FORMAT_NHWC);
    output0->AddInputDesc(tensorDesc2);
    output0->AddOutputDesc(tensorDesc2);
  }
  OpDescPtr squareOpDesc = std::make_shared<OpDesc>("squareop", "Square");
  {
    GeTensorDesc inputTensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({32, 32}));
    inputTensorDesc.SetOriginFormat(FORMAT_NHWC);

    GeTensorDesc outputTensorDesc(GeShape({32, 32}), FORMAT_NHWC, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({32, 32}));
    outputTensorDesc.SetOriginFormat(FORMAT_NHWC);
    squareOpDesc->AddInputDesc(inputTensorDesc);
    squareOpDesc->AddOutputDesc("x", outputTensorDesc);
  }
  OpDescPtr reduceSumDopDesc = std::make_shared<OpDesc>("reduceSumDop", "ReduceSumD");
  {
    GeTensorDesc inputTensorDesc(GeShape({64, 32}), FORMAT_NHWC, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({64, 32}));
    inputTensorDesc.SetOriginFormat(FORMAT_NHWC);
    GeTensorDesc outputTensorDesc(GeShape({64, 32}), FORMAT_NHWC, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({64, 32}));
    outputTensorDesc.SetOriginFormat(FORMAT_NHWC);
    reduceSumDopDesc->AddInputDesc("y", inputTensorDesc);
    reduceSumDopDesc->AddOutputDesc(outputTensorDesc);
    ge::AttrUtils::SetListInt(reduceSumDopDesc, "axes", {1, 1});
    ge::AttrUtils::SetBool(reduceSumDopDesc, "keep_dims", false);
  }
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("square_sum_v2_fusion_test");
  NodePtr inputdata_node = graph->AddNode(input0);
  NodePtr outputdata_node = graph->AddNode(output0);
  NodePtr square_node = graph->AddNode(squareOpDesc);
  NodePtr reducesumd_node = graph->AddNode(reduceSumDopDesc);
  GraphUtils::AddEdge(inputdata_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), reducesumd_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), outputdata_node->GetInDataAnchor(0));
  fe::FusionPassTestUtils::RunGraphFusionPass("SquareSumV2", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool square_sum_v2_fusion = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "SquareSumV2") {
      square_sum_v2_fusion = true;
    }
  }
  EXPECT_EQ(square_sum_v2_fusion, false);
}

// error test:sum node is second output node of square
TEST_F(square_sum_v2_fusion_test, square_sum_v2_fusion_test_dynamic_shape1) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  std::vector<int64_t> shape = {-1, 64};
  auto dtype = ge::DT_FLOAT;
  auto format = ge::FORMAT_ND;
  auto data = builder.AddNode("Data", "Data", 0, 1, shape, format, dtype);
  auto square_node = builder.AddNode("square", "Square", 1, 1, shape, format, dtype);
  auto reduce_sum_node = builder.AddNode("reduce_sum", "ReduceSumD", 1, 1, shape, format, dtype);
  auto net_output0 = builder.AddNode("NetOutput", "NetOutput", 1, 0, {1, 1}, format, dtype);
  auto net_output1 = builder.AddNode("NetOutput", "NetOutput", 1, 0, shape, format, dtype);

  ge::AttrUtils::SetListInt(reduce_sum_node->GetOpDesc(), "axes", {-1, -1});
  ge::AttrUtils::SetBool(reduce_sum_node->GetOpDesc(), "keep_dims", false);
  builder.AddDataEdge(data, 0, square_node, 0);
  builder.AddDataEdge(square_node, 0, net_output1, 0);
  builder.AddDataEdge(square_node, 0, reduce_sum_node, 0);
  builder.AddDataEdge(reduce_sum_node, 0, net_output0, 0);

  auto graph = builder.GetGraph();
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("SquareSumV2", fe::BUILT_IN_GRAPH_PASS, *graph);
  bool square_sum_v2_fusion = false;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "SquareSumV2") {
      square_sum_v2_fusion = true;
    }
  }
  EXPECT_EQ(square_sum_v2_fusion, false);
  EXPECT_NE(ret, SUCCESS);
}

}  // namespace fe
