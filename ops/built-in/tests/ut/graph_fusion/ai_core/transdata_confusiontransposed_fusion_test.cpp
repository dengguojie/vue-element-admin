/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fc_transdata_merge_fusion_pass_test.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"

using namespace ge;
using namespace op;

class transdata_confusiontransposed_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "transdata_confusiontransposed_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transdata_confusiontransposed_fusion_pass_test TearDown" << std::endl;
  }
};

// Test normal scene.
TEST_F(transdata_confusiontransposed_fusion_pass_test, transdata_confusiontransposed_fusion_pass_test_01) {
  std::string testCaseName = "transdata_confusiontransposed_fusion_pass_test_01";

  Data inputData = Data("data_as_MatMul_38");
  {
    TensorDesc inputDataDesc(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    inputDataDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    inputDataDesc.SetOriginFormat(FORMAT_NCHW);

    inputData.update_input_desc_x(inputDataDesc);
    inputData.update_output_desc_y(inputDataDesc);
  }

  auto transdata1 = op::TransData("trans_Transdata_25");
  {
    ge::TensorDesc transdata1InputTensorDesc(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    transdata1InputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transdata1InputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc transdata1OutputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    transdata1OutputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transdata1OutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    transdata1.update_input_desc_src(transdata1InputTensorDesc);
    transdata1.update_output_desc_dst(transdata1OutputTensorDesc);
    transdata1.set_attr_src_format("FRACTAL_NZ");
    transdata1.set_attr_dst_format("ND");
  }

  auto reformat = op::ReFormat("trans_ReFormat_26");
  {
    ge::TensorDesc reformatInputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    reformatInputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    reformatInputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc reformatOutputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_NCHW, DT_FLOAT16);
    reformatOutputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    reformatOutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    reformat.update_input_desc_x(reformatInputTensorDesc);
    reformat.update_output_desc_y(reformatOutputTensorDesc);
  }

  auto confusionTransposeD = op::ConfusionTransposeD("Reshape_62/ConfusionTranspose");
  {
    ge::TensorDesc transposeInputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    transposeInputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transposeInputTensorDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc transposeOutputTensorDesc(ge::Shape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    transposeOutputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transposeOutputTensorDesc.SetOriginFormat(FORMAT_ND);

    confusionTransposeD.update_input_desc_x(transposeInputTensorDesc);
    confusionTransposeD.update_output_desc_y(transposeOutputTensorDesc);
    confusionTransposeD.set_attr_perm({0, 2, 3, 1});
    confusionTransposeD.set_attr_shape({1000, 64, 5, 64});
    confusionTransposeD.set_attr_transpose_first(false);
  }

  auto transdata2 = op::TransData("trans_Transdata_36");
  {
    ge::TensorDesc transdata2InputTensorDesc(ge::Shape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    transdata2InputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transdata2InputTensorDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc transdata2OutputTensorDesc(ge::Shape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    transdata2OutputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transdata2OutputTensorDesc.SetOriginFormat(FORMAT_ND);

    transdata2.update_input_desc_src(transdata2InputTensorDesc);
    transdata2.update_output_desc_dst(transdata2OutputTensorDesc);
    transdata2.set_attr_src_format("ND");
    transdata2.set_attr_dst_format("FRACTAL_NZ");
  }

  auto x2Data = op::Data("data_as_MatMul_36");
  {
    ge::TensorDesc tensorDescX2(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDescX2.SetOriginShape(ge::Shape({1000, 64, 320}));
    tensorDescX2.SetOriginFormat(FORMAT_NCHW);

    x2Data.update_input_desc_x(tensorDescX2);
    x2Data.update_output_desc_y(tensorDescX2);
  }

  transdata1.set_input_src(inputData);
  reformat.set_input_x(transdata1);
  confusionTransposeD.set_input_x(reformat);
  transdata2.set_input_src(confusionTransposeD);

  auto endOp = op::BatchMatMulV2("MatMul_75");
  endOp.set_input_x1(transdata2);
  endOp.set_input_x2(x2Data);

  std::vector<Operator> inputs{inputData, x2Data};
  std::vector<Operator> outputs{endOp};

  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransDataConfusionTransposeDFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findConfusionTransposeD = false;
  bool findTranspose = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ConfusionTransposeD") {
      findConfusionTransposeD = true;
    } else if (node->GetType() == "Transpose") {
      findTranspose = true;
    }
  }
  EXPECT_EQ(findConfusionTransposeD, false);
  EXPECT_EQ(findTranspose, true);
}

// Test bad attribute configuration: perm.
TEST_F(transdata_confusiontransposed_fusion_pass_test, transdata_confusiontransposed_fusion_pass_test_02) {
  std::string testCaseName = "transdata_confusiontransposed_fusion_pass_test_02";

  Data inputData = Data("data_as_MatMul_38");
  {
    TensorDesc inputDataDesc(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    inputDataDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    inputDataDesc.SetOriginFormat(FORMAT_NCHW);

    inputData.update_input_desc_x(inputDataDesc);
    inputData.update_output_desc_y(inputDataDesc);
  }

  auto transdata1 = op::TransData("trans_Transdata_25");
  {
    ge::TensorDesc transdata1InputTensorDesc(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    transdata1InputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transdata1InputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc transdata1OutputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    transdata1OutputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transdata1OutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    transdata1.update_input_desc_src(transdata1InputTensorDesc);
    transdata1.update_output_desc_dst(transdata1OutputTensorDesc);
    transdata1.set_attr_src_format("FRACTAL_NZ");
    transdata1.set_attr_dst_format("ND");
  }

  auto reformat = op::ReFormat("trans_ReFormat_26");
  {
    ge::TensorDesc reformatInputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    reformatInputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    reformatInputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc reformatOutputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_NCHW, DT_FLOAT16);
    reformatOutputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    reformatOutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    reformat.update_input_desc_x(reformatInputTensorDesc);
    reformat.update_output_desc_y(reformatOutputTensorDesc);
  }

  auto confusionTransposeD = op::ConfusionTransposeD("Reshape_62/ConfusionTranspose");
  {
    ge::TensorDesc transposeInputTensorDesc(ge::Shape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    transposeInputTensorDesc.SetOriginShape(ge::Shape({1000, 64, 320}));
    transposeInputTensorDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc transposeOutputTensorDesc(ge::Shape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    transposeOutputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transposeOutputTensorDesc.SetOriginFormat(FORMAT_ND);

    confusionTransposeD.update_input_desc_x(transposeInputTensorDesc);
    confusionTransposeD.update_output_desc_y(transposeOutputTensorDesc);
    confusionTransposeD.set_attr_perm({0, 3, 2, 1});
    confusionTransposeD.set_attr_shape({1000, 64, 5, 64});
    confusionTransposeD.set_attr_transpose_first(false);
  }

  auto transdata2 = op::TransData("trans_Transdata_36");
  {
    ge::TensorDesc transdata2InputTensorDesc(ge::Shape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    transdata2InputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transdata2InputTensorDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc transdata2OutputTensorDesc(ge::Shape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    transdata2OutputTensorDesc.SetOriginShape(ge::Shape({1000, 5, 64, 64}));
    transdata2OutputTensorDesc.SetOriginFormat(FORMAT_ND);

    transdata2.update_input_desc_src(transdata2InputTensorDesc);
    transdata2.update_output_desc_dst(transdata2OutputTensorDesc);
    transdata2.set_attr_src_format("ND");
    transdata2.set_attr_dst_format("FRACTAL_NZ");
  }

  auto x2Data = op::Data("data_as_MatMul_36");
  {
    ge::TensorDesc tensorDescX2(ge::Shape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDescX2.SetOriginShape(ge::Shape({1000, 64, 320}));
    tensorDescX2.SetOriginFormat(FORMAT_NCHW);

    x2Data.update_input_desc_x(tensorDescX2);
    x2Data.update_output_desc_y(tensorDescX2);
  }

  transdata1.set_input_src(inputData);
  reformat.set_input_x(transdata1);
  confusionTransposeD.set_input_x(reformat);
  transdata2.set_input_src(confusionTransposeD);

  auto endOp = op::BatchMatMulV2("MatMul_75");
  endOp.set_input_x1(transdata2);
  endOp.set_input_x2(x2Data);

  std::vector<Operator> inputs{inputData, x2Data};
  std::vector<Operator> outputs{endOp};

  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransDataConfusionTransposeDFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findConfusionTransposeD = false;
  bool findTranspose = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ConfusionTransposeD") {
      findConfusionTransposeD = true;
    } else if (node->GetType() == "Transpose") {
      findTranspose = true;
    }
  }
  EXPECT_EQ(findConfusionTransposeD, true);
  EXPECT_EQ(findTranspose, false);
}

// Test control edge.
TEST_F(transdata_confusiontransposed_fusion_pass_test, transdata_confusiontransposed_fusion_pass_test_03) {
  std::string testCaseName = "transdata_confusiontransposed_fusion_pass_test_03";

  OpDescPtr inputDataOpDesc = std::make_shared<OpDesc>("data_as_MatMul_38", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    tensorDesc.SetOriginFormat(FORMAT_NCHW);

    inputDataOpDesc->AddInputDesc(tensorDesc);
    inputDataOpDesc->AddOutputDesc(tensorDesc);
  }

  OpDescPtr transdata1OpDesc = std::make_shared<OpDesc>("trans_Transdata_25", "TransData");
  {
    GeTensorDesc inputTensorDesc(GeShape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    inputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    GeTensorDesc outputTensorDesc(GeShape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    transdata1OpDesc->AddInputDesc(inputTensorDesc);
    transdata1OpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetStr(transdata1OpDesc, "src_format", "FRACTAL_NZ");
    AttrUtils::SetStr(transdata1OpDesc, "dst_format", "ND");
  }

  OpDescPtr reformatOpDesc = std::make_shared<OpDesc>("trans_ReFormat_26", "ReFormat");
  {
    GeTensorDesc inputTensorDesc(GeShape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    inputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    GeTensorDesc outputTensorDesc(GeShape({1000, 64, 320}), FORMAT_NCHW, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    reformatOpDesc->AddInputDesc(inputTensorDesc);
    reformatOpDesc->AddOutputDesc(outputTensorDesc);
  }

  OpDescPtr confusionTransposeDOpDesc = std::make_shared<OpDesc>("Reshape_62/ConfusionTranspose", "ConfusionTransposeD");
  {
    GeTensorDesc inputTensorDesc(GeShape({1000, 64, 320}), FORMAT_ND, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    inputTensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc outputTensorDesc(GeShape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    outputTensorDesc.SetOriginFormat(FORMAT_ND);

    confusionTransposeDOpDesc->AddInputDesc(inputTensorDesc);
    confusionTransposeDOpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetListInt(confusionTransposeDOpDesc, "perm", {0, 3, 2, 1});
    AttrUtils::SetListInt(confusionTransposeDOpDesc, "shape", {1000, 64, 5, 64});
    AttrUtils::SetBool(confusionTransposeDOpDesc, "transpose_first", false);
  }

  OpDescPtr transdata2OpDesc = std::make_shared<OpDesc>("trans_Transdata_36", "TransData");
  {
    GeTensorDesc inputTensorDesc(GeShape({1000, 5, 64, 64}), FORMAT_ND, DT_FLOAT16);
    inputTensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    inputTensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc outputTensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    outputTensorDesc.SetOriginFormat(FORMAT_ND);

    transdata2OpDesc->AddInputDesc(inputTensorDesc);
    transdata2OpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetStr(transdata2OpDesc, "src_format", "ND");
    AttrUtils::SetStr(transdata2OpDesc, "dst_format", "FRACTAL_NZ");
  }

  OpDescPtr x2DataOpDesc = std::make_shared<OpDesc>("data_as_MatMul_36", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({1000, 20, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({1000, 64, 320}));
    tensorDesc.SetOriginFormat(FORMAT_NCHW);

    x2DataOpDesc->AddInputDesc(tensorDesc);
    x2DataOpDesc->AddOutputDesc(tensorDesc);
  }

  OpDescPtr biasOpDesc = std::make_shared<OpDesc>("bias_0", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    tensorDesc.SetOriginFormat(FORMAT_ND);

    biasOpDesc->AddInputDesc(tensorDesc);
    biasOpDesc->AddOutputDesc(tensorDesc);
  }

  OpDescPtr addsOpDesc = std::make_shared<OpDesc>("adds_0", "Adds");
  {
    GeTensorDesc tensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    tensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    tensorDesc.SetOriginFormat(FORMAT_ND);

    addsOpDesc->AddInputDesc(tensorDesc);
    addsOpDesc->AddOutputDesc(tensorDesc);
    AttrUtils::SetFloat(addsOpDesc, "value", 1.0);
  }

  OpDescPtr endOpDesc = std::make_shared<OpDesc>("MatMul_75", "BatchMatMulV2");
  {
    GeTensorDesc input1TensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input1TensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    input1TensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc input2TensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input2TensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    input2TensorDesc.SetOriginFormat(FORMAT_ND);
  
    GeTensorDesc biasTensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    biasTensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    biasTensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc outputTensorDesc(GeShape({1000, 5, 4, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({1000, 5, 64, 64}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    endOpDesc->AddInputDesc("x1", input1TensorDesc);
    endOpDesc->AddInputDesc("x2", input2TensorDesc);
    endOpDesc->AddInputDesc("bias", biasTensorDesc);
    endOpDesc->AddOutputDesc(outputTensorDesc);
  }

  ge::ComputeGraphPtr computeGraph = std::make_shared<ComputeGraph>(testCaseName);
  NodePtr inputDataNode = computeGraph->AddNode(inputDataOpDesc);
  NodePtr transdata1Node = computeGraph->AddNode(transdata1OpDesc);
  NodePtr reformatNode = computeGraph->AddNode(reformatOpDesc);
  NodePtr confusionTransposeDNode = computeGraph->AddNode(confusionTransposeDOpDesc);
  NodePtr transdata2Node = computeGraph->AddNode(transdata2OpDesc);
  NodePtr x2DataNode = computeGraph->AddNode(x2DataOpDesc);
  NodePtr biasNode = computeGraph->AddNode(biasOpDesc);
  NodePtr addsNode = computeGraph->AddNode(addsOpDesc);
  NodePtr endDataNode = computeGraph->AddNode(endOpDesc);

  GraphUtils::AddEdge(inputDataNode->GetOutDataAnchor(0), transdata1Node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1Node->GetOutDataAnchor(0), reformatNode->GetInDataAnchor(0));
  GraphUtils::AddEdge(reformatNode->GetOutDataAnchor(0), confusionTransposeDNode->GetInDataAnchor(0));
  GraphUtils::AddEdge(confusionTransposeDNode->GetOutDataAnchor(0), transdata2Node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata2Node->GetOutDataAnchor(0), endDataNode->GetInDataAnchor(0));
  GraphUtils::AddEdge(x2DataNode->GetOutDataAnchor(0), endDataNode->GetInDataAnchor(1));
  GraphUtils::AddEdge(biasNode->GetOutDataAnchor(0), addsNode->GetInDataAnchor(0));
  GraphUtils::AddEdge(addsNode->GetOutDataAnchor(0), endDataNode->GetInDataAnchor(2));

  GraphUtils::AddEdge(addsNode->GetOutControlAnchor(), confusionTransposeDNode->GetInControlAnchor());

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransDataConfusionTransposeDFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findConfusionTransposeD = false;
  bool findTranspose = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ConfusionTransposeD") {
      findConfusionTransposeD = true;
    } else if (node->GetType() == "Transpose") {
      findTranspose = true;
    }
  }
  EXPECT_EQ(findConfusionTransposeD, true);
  EXPECT_EQ(findTranspose, false);
}
