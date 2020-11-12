/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file confusion_softmax_grad_fusion_pass.cpp
 * \brief confusion_softmax_grad fusion pass( --> confusion_softmax_grad)
 */
#include "confusion_softmax_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {

// node type
static const string NODE_SOFTMAXGRAD = "ConfusionSoftmaxGrad";
static const string NODE_MUL = "Mul";
static const string NODE_SUB = "Sub";
static const string NODE_REDUCESUMD = "ReduceSumD";

// node name id
static const string PATTERN_MUL = "mul";
static const string PATTERN_SUB = "sub";
static const string PATTERN_REDUCESUMD = "reducesumD";

// attr name
static const string ATTR_AXIS = "axes";

/*
before:
             input0
                   \
                    \
      input1 ————> Mul
         \        /
           \     /
             Sub
              |
              |
          reducesumD
              |
           output

after:
      input1  input0
          \      |
            \    |
         ConfusionSoftmaxGrad
                |
                |
              output
*/

vector<FusionPattern*> ConfusionSoftmaxGradFusionPass::DefinePatterns() {

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionMulGrad pattern begin");

  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConfusionSoftmaxGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  
  pattern->AddOpDesc(PATTERN_REDUCESUMD, {NODE_REDUCESUMD})
      .AddOpDesc(PATTERN_MUL, {NODE_MUL})
      .AddOpDesc(PATTERN_SUB, {NODE_SUB}) // mul->sub
      .SetInputs(PATTERN_REDUCESUMD, {PATTERN_SUB})// sub->reducesumD
      .SetOutput(PATTERN_REDUCESUMD);

  patterns.push_back(pattern);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionSoftmaxGradFusionPass pattern end");

  return patterns;
}


Status ConfusionSoftmaxGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionSoftmaxGradFusionPass fusion begin");

  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr reduceSumNode = GetNodeFromMapping(PATTERN_REDUCESUMD, mapping);

  FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(subNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sub node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(reduceSumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "reduce sum node is null, fusion failed."), return PARAM_INVALID);

  // check all nodes input size
  int inputSize = 0;
  inputSize = mulNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node size is [%lu], which not equal to 2.", inputSize),
    return NOT_CHANGED);

  inputSize = subNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node size is [%lu], which not equal to 2.", inputSize),
    return NOT_CHANGED);

  inputSize = reduceSumNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum node size is [%lu], which not equal to 1.", inputSize),
    return NOT_CHANGED);

  // check all nodes output size
  int outputSize = 0;
  outputSize = mulNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  outputSize = subNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  outputSize = reduceSumNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  if (mulNode->GetName().find("bert/encoder") == string::npos &&
      mulNode->GetName().find("loss/gradMul") == string::npos &&
      mulNode->GetName().find("loss-BertPretrainingLoss/gradMul") == string::npos) {
    return NOT_CHANGED;
  }

  // create softmax grad node op description
  std::string softmaxGradName = mulNode->GetName() + "/" + NODE_SOFTMAXGRAD;
  ge::OpDescPtr softmaxGradOpdesc = std::make_shared<ge::OpDesc>(softmaxGradName, NODE_SOFTMAXGRAD);
  FUSION_PASS_CHECK(softmaxGradOpdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "softmaxGradOpdesc is null, fusion failed."), return PARAM_INVALID);

  // define attrs of input edge based on orignal info
  ge::GeTensorDesc input0Ddesc = mulNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc input1Ddesc = subNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc outputDesc = reduceSumNode->GetOpDesc()->GetOutputDesc(0);
  softmaxGradOpdesc->AddInputDesc(0, input0Ddesc);
  softmaxGradOpdesc->AddInputDesc(1, input1Ddesc);
  softmaxGradOpdesc->AddOutputDesc(0, outputDesc);

  // add prelu node into graph
  ge::NodePtr softmaxGradNode = graph.AddNode(softmaxGradOpdesc);
  softmaxGradNode->GetOpDesc()->SetType(NODE_SOFTMAXGRAD);
  newNodes.push_back(softmaxGradNode);

  // add input edge
  ge::OutDataAnchorPtr input0Anchor = mulNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr input1Anchor = subNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::GraphUtils::AddEdge(input0Anchor, softmaxGradNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(input1Anchor, softmaxGradNode->GetInDataAnchor(1));

  // add output edge
  for (auto &inDataAnchor : reduceSumNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(reduceSumNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxGradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // check axis -1
  std::vector<int64_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(reduceSumNode->GetOpDesc(), ATTR_AXIS, axis),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get axis attr failed"), return FAILED);
  FUSION_PASS_CHECK(axis.size() != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum attr axis size is [%d], which not equal to 1.", axis.size()),
    return NOT_CHANGED);
  
  FUSION_PASS_CHECK(axis[0] != -1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum attr axis 0 is [%d], which not equal to -1.", axis[0]),
    return NOT_CHANGED);
  
  // remove useless node in subgraph
  FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(subNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sub node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reduceSumNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove reduce sum node failed."),
                    return FAILED);

  return SUCCESS;
}

REGISTER_PASS("ConfusionSoftmaxGradFusionPass", BUILT_IN_GRAPH_PASS, ConfusionSoftmaxGradFusionPass);

}  // namespace fe
