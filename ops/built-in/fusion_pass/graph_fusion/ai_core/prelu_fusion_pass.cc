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
 * \file prelu_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#include "prelu_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace std;

namespace fe {

// node type
static const string PRELU_NODE = "PRelu";
static const string RELU_NODE = "Relu";
static const string NEG_NODE = "Neg";
static const string MUL_NODE = "Mul";
static const string ADD_NODE = "Add";

// node name id
static const string PATTERN_RELU = "Relu";
static const string PATTERN_NEG = "Neg";
static const string PATTERN_RELU1 = "Relu_1";
static const string PATTERN_NEG1 = "Neg_1";
static const string PATTERN_MUL = "mul";
static const string PATTERN_ADD = "add";

/*
before:
             Conv2D
            /      \
           /        \
          /         Neg1
         /            /
        /    Neg    Relu1
       /        \    /
        \        Mul1
        Relu     /
           \    /
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

vector<FusionPattern*> PReluFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("PReluFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RELU, {RELU_NODE})
      .AddOpDesc(PATTERN_NEG, {NEG_NODE})
      .AddOpDesc(PATTERN_RELU1, {RELU_NODE})
      .AddOpDesc(PATTERN_NEG1, {NEG_NODE})
      .AddOpDesc(PATTERN_MUL, {MUL_NODE})
      .AddOpDesc(PATTERN_ADD, {ADD_NODE})
      .SetInputs(PATTERN_RELU1, {PATTERN_NEG1})              // Neg_1->Relu_1
      .SetInputs(PATTERN_MUL, {PATTERN_NEG, PATTERN_RELU1})  // Neg->mul,  Relu_1->mul
      .SetInputs(PATTERN_ADD, {PATTERN_RELU, PATTERN_MUL})   // mul->add, Relu->add
      .SetOutput(PATTERN_ADD);

  patterns.push_back(pattern);

  return patterns;
}

Status PReluFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion PReluFusionPass!");

  ge::NodePtr addNode = GetNodeFromMapping(PATTERN_ADD, mapping);
  ge::NodePtr reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);
  ge::NodePtr negNode = GetNodeFromMapping(PATTERN_NEG, mapping);
  ge::NodePtr relu1Node = GetNodeFromMapping(PATTERN_RELU1, mapping);
  ge::NodePtr neg1Node = GetNodeFromMapping(PATTERN_NEG1, mapping);
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);

  FUSION_PASS_CHECK(addNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "addNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(reluNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reluNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(negNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "negNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(relu1Node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu1Node is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(neg1Node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "neg1Node is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return PARAM_INVALID);

  // check all nodes input size
  int inputSize = 0;
  inputSize = reluNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = negNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "neg node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = relu1Node->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu1 node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = neg1Node->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "neg1 node size is [%lu], which not equal to 1.", inputSize),
                    return NOT_CHANGED);

  inputSize = mulNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node size is [%lu], which not equal to 2.", inputSize),
                    return NOT_CHANGED);

  inputSize = addNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add node size is [%lu], which not equal to 2.", inputSize),
                    return NOT_CHANGED);

  // check all nodes output size
  int outputSize = 0;
  outputSize = reluNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node output size is [%d], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  outputSize = negNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "neg node output size is [%d], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  outputSize = relu1Node->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "relu1 node output size is [%d], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  outputSize = neg1Node->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "neg1 node output size is [%d], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  outputSize = mulNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node output size is [%d], which not equal to 1.", outputSize),
                    return NOT_CHANGED);

  // check Neg1 and Relu node input are from the same Conv2D node
  OutDataAnchorPtr neg1InputAnchor = neg1Node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(neg1InputAnchor == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "neg1 node input anchor is null"),
                    return NOT_CHANGED);
  string neg1InputName = neg1InputAnchor->GetOwnerNode()->GetName();

  OutDataAnchorPtr reluInputAnchor = reluNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(reluInputAnchor == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node input anchor is null"),
                    return NOT_CHANGED);
  string reluInputName = reluInputAnchor->GetOwnerNode()->GetName();

  FUSION_PASS_CHECK(neg1InputName != reluInputName,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "neg1 node input name %s must equal to relu node input name %s.",
                            neg1InputName.c_str(), reluInputName.c_str()),
                    return NOT_CHANGED);

  // create prelu node op description
  std::string preluNodeName = addNode->GetName() + "_" + "prelu";
  std::shared_ptr<ge::OpDesc> preluOpdesc = std::make_shared<ge::OpDesc>(preluNodeName, PRELU_NODE);
  FUSION_PASS_CHECK(preluOpdesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "preluOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc neg1InputDesc = neg1Node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddInputDesc(0, neg1InputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add neg1Node input desc failed."), return FAILED);

  ge::GeTensorDesc negInputDesc = negNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddInputDesc(1, negInputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add negNode input desc failed."), return FAILED);

  ge::GeTensorDesc addOutputDesc = addNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(preluOpdesc->AddOutputDesc(addOutputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add addNode output desc failed."), return FAILED);

  // add prelu node into graph
  ge::NodePtr preluNode = graph.AddNode(preluOpdesc);
  preluNode->GetOpDesc()->SetType(PRELU_NODE);
  newNodes.push_back(preluNode);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(neg1Node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            preluNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between neg1Node and preluNode failed."), return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(negNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            preluNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between negNode and preluNode failed."), return FAILED);

  for (auto &inDataAnchor : addNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(addNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preluNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(addNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove addNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reluNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reluNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(negNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove negNode failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(relu1Node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove relu1Node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(neg1Node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove neg1Node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mulNode failed."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave graph fusion PReluFusionPass!");

  return SUCCESS;
}

REGISTER_PASS("PReluFusionPass", BUILT_IN_GRAPH_PASS, PReluFusionPass);

}  // namespace fe
