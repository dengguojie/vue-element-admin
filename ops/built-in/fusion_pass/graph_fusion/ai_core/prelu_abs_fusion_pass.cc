/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

/*!
 * \file prelu_abs_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#include "prelu_abs_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace std;

namespace fe {

// node type
static const string PRELU_NODE = "PRelu";
static const string RELU_NODE = "Relu";
static const string ABS_NODE = "Abs";
static const string SUB_NODE = "Sub";
static const string MUL_NODE = "Mul";
static const string ADD_NODE = "Add";

// node name id
static const string PATTERN_RELU = "Relu";
static const string PATTERN_ABS = "Abs";
static const string PATTERN_SUB = "sub";
static const string PATTERN_MUL1 = "mul_1";
static const string PATTERN_MUL = "mul";
static const string PATTERN_ADD = "add";
static const std::string PATTERN_INPUTS = "input";

/*
before:
             Conv2D
            /   |   \
           /    |    \
          /     |   Abs
         /      |   /
        /       Sub
       /         \
        \        Mul1
        Relu     /
           \   Mul
            \  /
            Add
             |
          MaxPool

 after:
             Conv2D
                |
                |
              PRelu
                |
             MaxPool
*/

vector<FusionPattern*> PReluAbsFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("PReluAbsFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RELU, {RELU_NODE})
      .AddOpDesc(PATTERN_ABS, {ABS_NODE})
      .AddOpDesc(PATTERN_SUB, {SUB_NODE})
      .AddOpDesc(PATTERN_MUL1, {MUL_NODE})
      .AddOpDesc(PATTERN_MUL, {MUL_NODE})
      .AddOpDesc(PATTERN_ADD, {ADD_NODE})
      .AddOpDesc(PATTERN_INPUTS)
      .SetInputs(PATTERN_SUB, {PATTERN_INPUTS, PATTERN_ABS})   // Abs->Sub
      .SetInputs(PATTERN_MUL, {PATTERN_INPUTS, PATTERN_SUB})   // Sub->mul
      .SetInputs(PATTERN_MUL1, {PATTERN_MUL, PATTERN_INPUTS})  // mul->mul1
      .SetInputs(PATTERN_ADD, {PATTERN_RELU, PATTERN_MUL1})    // mul1->add, Relu->add
      .SetOutput(PATTERN_ADD);

  patterns.push_back(pattern);

  return patterns;
}

Status PReluAbsFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion PReluAbsFusionPass!");

  ge::NodePtr addNode = GetNodeFromMapping(PATTERN_ADD, mapping);
  ge::NodePtr reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);
  ge::NodePtr absNode = GetNodeFromMapping(PATTERN_ABS, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr mul1Node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(addNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "addNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reluNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reluNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(absNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "absNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(subNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "subNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1Node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul1Node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mulNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return PARAM_INVALID);

  int inputSize = reluNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  int outputSize = reluNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node output size is [%lu], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  inputSize = absNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "abs node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = mul1Node->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul1 node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = subNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  outputSize = subNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node output size is [%lu], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  inputSize = mulNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node size is [%lu], which not equal to 2.", inputSize),
                    return NOT_CHANGED);

  outputSize = mulNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node output size is [%lu], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  inputSize = addNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add node size is [%lu], which not equal to 2.", inputSize),
                    return NOT_CHANGED);

  std::string preluNodeName = addNode->GetName() + "_" + "prelu";
  std::shared_ptr<ge::OpDesc> preluOpdesc = std::make_shared<ge::OpDesc>(preluNodeName, PRELU_NODE);
  FUSION_PASS_CHECK(preluOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "preluOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc reluInputDesc = reluNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddInputDesc("x", reluInputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reluNode input desc failed."),
                    return FAILED);

  ge::GeTensorDesc mulInputDesc = mulNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddInputDesc("weight", mulInputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulNode input desc failed."),
                    return FAILED);

  ge::GeTensorDesc addOutputDesc = addNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddOutputDesc("y", addOutputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add addNode output desc failed."),
                    return FAILED);

  ge::NodePtr preluNode = graph.AddNode(preluOpdesc);
  preluNode->GetOpDesc()->SetType(PRELU_NODE);
  newNodes.push_back(preluNode);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reluNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            preluNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge between reluNode and preluNode failed."), return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            preluNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge between mulNode and preluNode failed."), return FAILED);

  for (auto &inDataAnchor : addNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(addNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preluNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(addNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove addNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reluNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reluNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(subNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove subNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1Node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1Node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(absNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove absNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mulNode failed."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave graph fusion PReluAbsFusionPass!");

  return SUCCESS;
}

REGISTER_PASS("PReluAbsFusionPass", BUILT_IN_GRAPH_PASS, PReluAbsFusionPass);

}  // namespace fe
