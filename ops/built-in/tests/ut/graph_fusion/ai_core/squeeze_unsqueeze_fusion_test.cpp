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

#include <stdlib.h>
#include <nlohmann/json.hpp>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "selection_ops.h"
#include "split_combination_ops.h"
#include "vector_search.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class squeeze_unsqueeze_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "squeeze_unsqueeze_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "squeeze_unsqueeze_fusion_test TearDown" << std::endl;
  }
};

void RunSimpleTest(const std::string& testCaseName, const std::vector<int64_t>& squeezeInputDims,
                   const std::vector<int64_t>& squeezeOutputDims, const std::vector<int64_t>& squeezeAxis,
                   const std::vector<int64_t>& unsqueezeInputDims, const std::vector<int64_t>& unsqueezeOutputDims,
                   const std::vector<int64_t>& unsqueezeAxes, const bool& expectFlag) {
  Data inputData = Data("data_as_MaxPoolV3_41_82");
  {
    TensorDesc inputDataDesc(ge::Shape(squeezeInputDims), FORMAT_NCHW, DT_FLOAT);
    inputDataDesc.SetOriginShape(ge::Shape(squeezeInputDims));
    inputDataDesc.SetOriginFormat(FORMAT_NCHW);

    inputData.update_input_desc_x(inputDataDesc);
    inputData.update_output_desc_y(inputDataDesc);
  }

  auto squeeze = op::Squeeze("PartitionedCall_SqueezeMaxpoolv3_83");
  {
    ge::TensorDesc squeezeInputTensorDesc(ge::Shape(squeezeInputDims), FORMAT_NCHW, DT_FLOAT);
    squeezeInputTensorDesc.SetOriginShape(ge::Shape(squeezeInputDims));
    squeezeInputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc squeezeOutputTensorDesc(ge::Shape(squeezeOutputDims), FORMAT_NCHW, DT_FLOAT);
    squeezeOutputTensorDesc.SetOriginShape(ge::Shape(squeezeOutputDims));
    squeezeOutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    squeeze.update_input_desc_x(squeezeInputTensorDesc);
    squeeze.update_output_desc_y(squeezeOutputTensorDesc);
    squeeze.set_attr_axis(squeezeAxis);
  }

  auto unsqueeze = op::Unsqueeze("PartitionedCall_UnsqueezeX_76");
  {
    ge::TensorDesc unsqueezeInputTensorDesc(ge::Shape(unsqueezeInputDims), FORMAT_NCHW, DT_FLOAT);
    unsqueezeInputTensorDesc.SetOriginShape(ge::Shape(unsqueezeInputDims));
    unsqueezeInputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc unsqueezeOutputTensorDesc(ge::Shape(unsqueezeOutputDims), FORMAT_NCHW, DT_FLOAT);
    unsqueezeOutputTensorDesc.SetOriginShape(ge::Shape(unsqueezeOutputDims));
    unsqueezeOutputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    unsqueeze.update_input_desc_x(unsqueezeInputTensorDesc);
    unsqueeze.update_output_desc_y(unsqueezeOutputTensorDesc);
    unsqueeze.set_attr_axes(unsqueezeAxes);
  }

  auto constOp1 = op::Const("dynamic_const_25818_29");
  constOp1.set_attr_value(Tensor(TensorDesc(ge::Shape({160, 160, 1, 3}), FORMAT_ND, DT_FLOAT)));
  constOp1.update_output_desc_y(TensorDesc(ge::Shape({160, 160, 1, 3}), FORMAT_ND, DT_FLOAT));

  auto constOp2 = op::Const("sequenceModeling.conv2.bias_199");
  constOp2.set_attr_value(Tensor(TensorDesc(ge::Shape({160}), FORMAT_ND, DT_FLOAT)));
  constOp2.update_output_desc_y(TensorDesc(ge::Shape({160}), FORMAT_ND, DT_FLOAT));

  squeeze.set_input_x(inputData);
  unsqueeze.set_input_x(squeeze);

  auto endOp = op::Conv2D("PartitionedCall_Conv2D_40_78");
  endOp.set_input_x(unsqueeze)
      .set_input_filter(constOp1)
      .set_input_bias(constOp2)
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_pads({0, 0, 0, 0})
      .set_attr_dilations({1, 1, 1, 1})
      .set_attr_groups(1)
      .set_attr_offset_x(0);

  std::vector<Operator> inputs{inputData, constOp1, constOp2};
  std::vector<Operator> outputs{endOp};

  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("SqueezeUnsqueezeFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool flag = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "Squeeze" || node->GetType() == "Unsqueeze") {
      flag = true;
    }
  }
  EXPECT_EQ(flag, expectFlag);
}

// Test normal scene.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_01) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_01", {8, 160, 63, 1}, {8, 160, 63}, {3}, {8, 160, 63}, {8, 160, 1, 63},
                {2}, false);
}

// Test normal scene.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_02) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_02", {8, 160, 1, 63}, {8, 160, 63}, {2}, {8, 160, 63}, {8, 160, 63, 1},
                {3}, false);
}

// Test normal scene.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_03) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_03", {8, 160, 1, 63}, {8, 160, 63}, {2}, {8, 160, 63}, {8, 160, 1, 63},
                {2}, false);
}

// Test normal scene.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_04) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_04", {8, 160, 63, 1}, {8, 160, 63}, {3}, {8, 160, 63}, {8, 160, 63, 1},
                {3}, false);
}

// Test bad shape.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_05) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_05", {8, 160, 63}, {8, 160, 63}, {}, {8, 160, 63}, {8, 160, 1, 63, 1, 1},
                {2}, true);
}

// Test bad shape.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_06) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_06", {8, 160, 63, 1}, {8, 160, 63}, {3}, {888, 160, 63},
                {888, 160, 1, 63}, {2}, true);
}

// Test bad shape.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_07) {
  RunSimpleTest("squeeze_unsqueeze_fusion_test_07", {8, 1, 63, 63}, {8, 160, 63}, {1}, {8, 160, 63}, {8, 1, 1, 63}, {2},
                true);
}

// Test control edge.
TEST_F(squeeze_unsqueeze_fusion_test, squeeze_unsqueeze_fusion_test_08) {
  std::string testCaseName = "squeeze_unsqueeze_fusion_test_08";

  OpDescPtr inputDataOpDesc = std::make_shared<OpDesc>("data_as_MaxPoolV3_41_82", "Data");
  {
    GeTensorDesc tensorDesc(GeShape({8, 160, 63, 1}), FORMAT_NCHW, DT_FLOAT);
    tensorDesc.SetOriginShape(GeShape({8, 160, 63, 1}));
    tensorDesc.SetOriginFormat(FORMAT_NCHW);

    inputDataOpDesc->AddInputDesc(tensorDesc);
    inputDataOpDesc->AddOutputDesc(tensorDesc);
  }

  OpDescPtr squeezeOpDesc = std::make_shared<OpDesc>("PartitionedCall_SqueezeMaxpoolv3_83", "Squeeze");
  {
    GeTensorDesc inputTensorDesc(GeShape({8, 160, 63, 1}), FORMAT_NCHW, DT_FLOAT);
    inputTensorDesc.SetOriginShape(GeShape({8, 160, 63, 1}));
    inputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    GeTensorDesc outputTensorDesc(GeShape({8, 160, 63}), FORMAT_NCHW, DT_FLOAT);
    outputTensorDesc.SetOriginShape(GeShape({8, 160, 63}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    squeezeOpDesc->AddInputDesc(inputTensorDesc);
    squeezeOpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetListInt(squeezeOpDesc, "axis", {3});
  }

  OpDescPtr unsqueezeOpDesc = std::make_shared<OpDesc>("PartitionedCall_UnsqueezeX_76", "Unsqueeze");
  {
    GeTensorDesc inputTensorDesc(GeShape({8, 160, 63}), FORMAT_NCHW, DT_FLOAT);
    inputTensorDesc.SetOriginShape(GeShape({8, 160, 63}));
    inputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    GeTensorDesc outputTensorDesc(GeShape({8, 160, 1, 63}), FORMAT_NCHW, DT_FLOAT);
    outputTensorDesc.SetOriginShape(GeShape({8, 160, 1, 63}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    unsqueezeOpDesc->AddInputDesc(inputTensorDesc);
    unsqueezeOpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetListInt(unsqueezeOpDesc, "axis", {2});
  }

  OpDescPtr filterOpDesc = std::make_shared<OpDesc>("dynamic_const_25818_29", "Const");
  AttrUtils::SetTensor(filterOpDesc, "value", GeTensor(GeTensorDesc(GeShape({160, 160, 1, 3}), FORMAT_ND, DT_FLOAT)));
  filterOpDesc->AddOutputDesc(GeTensorDesc(GeShape({160, 160, 1, 3}), FORMAT_ND, DT_FLOAT));

  OpDescPtr biasOpDesc = std::make_shared<OpDesc>("sequenceModeling.conv2.bias_199", "Const");
  AttrUtils::SetTensor(biasOpDesc, "value", GeTensor(GeTensorDesc(GeShape({160}), FORMAT_ND, DT_FLOAT)));
  biasOpDesc->AddOutputDesc(GeTensorDesc(GeShape({160}), FORMAT_ND, DT_FLOAT));

  OpDescPtr addsOpDesc = std::make_shared<OpDesc>("adds_0", "Adds");
  {
    GeTensorDesc tensorDesc(GeShape({160}), FORMAT_ND, DT_FLOAT);
    tensorDesc.SetOriginShape(GeShape({160}));
    tensorDesc.SetOriginFormat(FORMAT_ND);

    addsOpDesc->AddInputDesc(tensorDesc);
    addsOpDesc->AddOutputDesc(tensorDesc);
    AttrUtils::SetFloat(addsOpDesc, "value", 1.0);
  }

  OpDescPtr endOpDesc = std::make_shared<OpDesc>("PartitionedCall_Conv2D_40_78", "Conv2D");
  {
    GeTensorDesc xTensorDesc(GeShape({8, 160, 1, 63}), FORMAT_NCHW, DT_FLOAT);
    xTensorDesc.SetOriginShape(GeShape({8, 160, 1, 63}));
    xTensorDesc.SetOriginFormat(FORMAT_NCHW);

    GeTensorDesc filterTensorDesc(GeShape({160, 160, 1, 3}), FORMAT_ND, DT_FLOAT);
    filterTensorDesc.SetOriginShape(GeShape({160, 160, 1, 3}));
    filterTensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc biasTensorDesc(GeShape({160}), FORMAT_ND, DT_FLOAT);
    biasTensorDesc.SetOriginShape(GeShape({160}));
    biasTensorDesc.SetOriginFormat(FORMAT_ND);

    GeTensorDesc outputTensorDesc(GeShape({160, 160, 1, 3}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    outputTensorDesc.SetOriginShape(GeShape({160, 160, 1, 3}));
    outputTensorDesc.SetOriginFormat(FORMAT_NCHW);

    endOpDesc->AddInputDesc("x", xTensorDesc);
    endOpDesc->AddInputDesc("filter", filterTensorDesc);
    endOpDesc->AddInputDesc("bias", biasTensorDesc);
    endOpDesc->AddOutputDesc(outputTensorDesc);
    AttrUtils::SetListInt(endOpDesc, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(endOpDesc, "pads", {0, 0, 0, 0});
    AttrUtils::SetListInt(endOpDesc, "dilations", {1, 1, 1, 1});
    AttrUtils::SetInt(endOpDesc, "groups", 1);
    AttrUtils::SetInt(endOpDesc, "offset_x", 0);
  }

  ge::ComputeGraphPtr computeGraph = std::make_shared<ComputeGraph>(testCaseName);
  {
    NodePtr inputDataNode = computeGraph->AddNode(inputDataOpDesc);
    NodePtr squeezeNode = computeGraph->AddNode(squeezeOpDesc);
    NodePtr unsqueezeNode = computeGraph->AddNode(unsqueezeOpDesc);

    NodePtr filterNode = computeGraph->AddNode(filterOpDesc);

    NodePtr biasNode = computeGraph->AddNode(biasOpDesc);
    NodePtr addsNode = computeGraph->AddNode(addsOpDesc);

    NodePtr endNode = computeGraph->AddNode(endOpDesc);

    GraphUtils::AddEdge(inputDataNode->GetOutDataAnchor(0), squeezeNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(squeezeNode->GetOutDataAnchor(0), unsqueezeNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(unsqueezeNode->GetOutDataAnchor(0), endNode->GetInDataAnchor(0));

    GraphUtils::AddEdge(filterNode->GetOutDataAnchor(0), endNode->GetInDataAnchor(1));

    GraphUtils::AddEdge(biasNode->GetOutDataAnchor(0), addsNode->GetInDataAnchor(0));
    GraphUtils::AddEdge(addsNode->GetOutDataAnchor(0), endNode->GetInDataAnchor(2));

    GraphUtils::AddEdge(addsNode->GetOutControlAnchor(), unsqueezeNode->GetInControlAnchor());
  }

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("SqueezeUnsqueezeFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool flag = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "Squeeze" || node->GetType() == "Unsqueeze") {
      flag = true;
    }
  }
  EXPECT_EQ(flag, true);
}
