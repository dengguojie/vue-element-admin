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
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace std;
using namespace ge;

namespace fe {
// node type
static const string TYPE_SOFTMAXGRAD = "ConfusionSoftmaxGrad";
static const string TYPE_MUL = "Mul";
static const string TYPE_SUB = "Sub";
static const string TYPE_REDUCESUMD = "ReduceSumD";

// node name id
static const string PATTERN_MUL = "Mul";
static const string PATTERN_SUB = "Sub";
static const string PATTERN_SUM = "Sum";

// attr name
static const string ATTR_AXIS = "axes";

/*
before:
  input0    input1
    |  \      |
    |   \———>Mul
    |         |
    |      reducesumD (attr axis=-1)
    |         |
    |——————> Sub
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
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionSoftmaxGrad pattern begin.");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConfusionSoftmaxGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_MUL, {TYPE_MUL})
      .AddOpDesc(PATTERN_SUM, {TYPE_REDUCESUMD})
      .AddOpDesc(PATTERN_SUB, {TYPE_SUB})
      .SetInputs(PATTERN_SUM, {PATTERN_MUL}) // mul->reducesumD
      .SetInputs(PATTERN_SUB, {PATTERN_SUM}) // reducesumD->sub
      .SetOutput(PATTERN_SUB);

  patterns.push_back(pattern);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionSoftmaxGradFusionPass pattern end.");

  return patterns;
}

Status ConfusionSoftmaxGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter ConfusionSoftmaxGradFusionPass.");

  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr reduceSumNode = GetNodeFromMapping(PATTERN_SUM, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);

  FUSION_PASS_CHECK(mulNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reduceSumNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reduce sum node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(subNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sub node is null, fusion failed."),
                    return PARAM_INVALID);
  NOT_CHANGED_WITH_DYNAMIC_NODE({mulNode}); // dynamic not changed

  // check all nodes input size
  int inputSize = 0;
  inputSize = mulNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node size is [%lu], which not equal to 2.", inputSize),
    return NOT_CHANGED);

  inputSize = reduceSumNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum node size is [%lu], which not equal to 1.", inputSize),
    return NOT_CHANGED);

  inputSize = subNode->GetInDataNodes().size();
  FUSION_PASS_CHECK(inputSize != 2,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node size is [%lu], which not equal to 2.", inputSize),
    return NOT_CHANGED);

  // check all nodes output size
  int outputSize = 0;
  outputSize = mulNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  outputSize = reduceSumNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  outputSize = subNode->GetOutDataNodes().size();
  FUSION_PASS_CHECK(outputSize != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node output size is [%d], which not equal to 1.", outputSize),
    return NOT_CHANGED);

  // check mul and sub node input0 are the same
  OutDataAnchorPtr mulInput0Anchor = mulNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(mulInput0Anchor == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node input anchor is null"),  return NOT_CHANGED);
  string mulInput0Name = mulInput0Anchor->GetOwnerNode()->GetName();

  OutDataAnchorPtr subInput0Anchor = subNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(subInput0Anchor == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub node input anchor is null"), return NOT_CHANGED);
  string subInput0Name = subInput0Anchor->GetOwnerNode()->GetName();

  FUSION_PASS_CHECK(mulInput0Name != subInput0Name,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node input name %s must equal to sub node input name %s.",
                            mulInput0Name.c_str(), subInput0Name.c_str()), return NOT_CHANGED);

  // check axis -1
  std::vector<int64_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(reduceSumNode->GetOpDesc(), ATTR_AXIS, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get axis attr failed"), return FAILED);
  FUSION_PASS_CHECK(axis.size() != 1,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum attr axis size is [%d], which not equal to 1.", axis.size()),
    return NOT_CHANGED);
  int64_t axisValue0 = axis[0];
  OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum axisValue0=%d", axisValue0);

  ge::GeTensorDesc reduceSumInputDesc = reduceSumNode->GetOpDesc()->GetInputDesc(0);
  ge::GeShape reduceSumInputShape = reduceSumInputDesc.GetShape();
  auto reduceSumInputDimNum = reduceSumInputShape.GetDims().size();
  axisValue0 = axisValue0 < 0 ? axisValue0 + int64_t(reduceSumInputDimNum) : axisValue0;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum reduceSumInputDimNum=%d", reduceSumInputDimNum);

  // input is scalar
  FUSION_PASS_CHECK(reduceSumInputDimNum == 0,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce sum reduceSumInputDimNum is %d.", reduceSumInputDimNum),
    return NOT_CHANGED);

  // axis is not the last -1
  FUSION_PASS_CHECK((axisValue0 % reduceSumInputDimNum) != (reduceSumInputDimNum - 1),
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "reduce sum axisValue0 % reduceSumInputDimNum is [%d], which not equal to reduceSumInputDimNum - 1 [%d].",
    (axisValue0 % reduceSumInputDimNum), (reduceSumInputDimNum - 1)), return NOT_CHANGED);

  // create softmax grad node op description
  std::string softmaxGradName = mulNode->GetName() + "/" + TYPE_SOFTMAXGRAD;
  ge::OpDescPtr softmaxGradOpdesc = std::make_shared<ge::OpDesc>(softmaxGradName, TYPE_SOFTMAXGRAD);
  FUSION_PASS_CHECK(softmaxGradOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "softmaxGradOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  // define attrs of input edge based on orignal info
  ge::GeTensorDesc input0Ddesc = mulNode->GetOpDesc()->GetInputDesc(0);
  softmaxGradOpdesc->AddInputDesc(0, input0Ddesc);

  ge::GeTensorDesc input1Ddesc = mulNode->GetOpDesc()->GetInputDesc(1);
  softmaxGradOpdesc->AddInputDesc(1, input1Ddesc);

  ge::GeTensorDesc outputDesc = subNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(softmaxGradOpdesc->AddOutputDesc((outputDesc)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add softmaxGrad output desc failed."), return FAILED);

  // add prelu node into graph
  ge::NodePtr softmaxGradNode = graph.AddNode(softmaxGradOpdesc);
  softmaxGradNode->GetOpDesc()->SetType(TYPE_SOFTMAXGRAD);
  newNodes.push_back(softmaxGradNode);

  // add input edge
  ge::OutDataAnchorPtr input0Anchor = mulNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr input1Anchor = mulNode->GetInDataAnchor(1)->GetPeerOutAnchor();
  ge::GraphUtils::AddEdge(input0Anchor, softmaxGradNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(input1Anchor, softmaxGradNode->GetInDataAnchor(1));

  // add output edge
  for (auto inDataAnchor : subNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(subNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxGradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add out data edge failed."), return FAILED);
  }

  // remove useless node in subgraph
  FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(subNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sub node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reduceSumNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reduce sum node failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Leave ConfusionSoftmaxGradFusionPass");
  return SUCCESS;
}

REGISTER_PASS("ZConfusionSoftmaxGradFusionPass", BUILT_IN_GRAPH_PASS, ConfusionSoftmaxGradFusionPass);
}  // namespace fe
