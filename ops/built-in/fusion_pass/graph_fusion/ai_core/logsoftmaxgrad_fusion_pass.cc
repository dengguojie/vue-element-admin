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
 * \file logsoftmaxgrad_fusion_pass.cpp
 * \brief logsoftmaxgrad fusion pass
 */
#include <string>
#include <vector>

#include "logsoftmaxgrad_fusion_pass.h"

#include "framework/common/string_util.h"
#include "external/graph/operator_factory.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

#include "op_log.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace fe {

static const char PATTERN_MUL[] = "mul";
static const char PATTERN_SUB[] = "sub";
static const char PATTERN_EXP[] = "exp";
static const char PATTERN_SUM[] = "sum";
static const char PATTERN_INPUT[] = "input";
static const char AXIS[] = "axis";
static const char LOGSOFTMAXGRAD[] = "LogSoftmaxGrad";
static const char ATTR_NAME_CONST[] = "axes";
static const uint8_t SUB_INPUT_NODE_NUM = 2;
static const uint8_t EXP_INPUT_NODE_NUM = 1;
static const uint8_t SUM_INPUT_NODE_NUM = 1;
static const uint8_t SUM_OUTPUT_NODE_NUM = 1;
static const uint8_t MUL_OUTPUT_NODE_NUM = 1;

static const char* SUM = "Sum";
static const char* SUMD = "ReduceSumD";
static const char* SUB = "Sub";
static const char* MUL = "Mul";
static const char* EXP = "Exp";

/*
        fusion pattern
                    sum---->mul---->sub
                             ^       ^
                             |       |
                             |       |
                            exp    input
 */
vector<FusionPattern*> LogSoftmaxGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LogSoftmaxGradFusion");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern not success!");
    return patterns;
  }

  pattern->AddOpDesc(PATTERN_EXP, {EXP})
      .AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_SUB, {SUB})
      .AddOpDesc(PATTERN_SUM, {SUM, SUMD})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_MUL, {PATTERN_SUM, PATTERN_EXP})
      .SetInputs(PATTERN_SUB, {PATTERN_INPUT, PATTERN_MUL})
      .SetOutput(PATTERN_SUB);
  patterns.push_back(pattern);
  return patterns;
}

Status LogSoftmaxGradFusionPass::IsMatch(ge::NodePtr sumNode, ge::NodePtr subNode, ge::NodePtr expNode,
                                         ge::NodePtr mulNode) {
  auto subInputDataNodes = subNode->GetInDataNodes();
  auto sumInputDataNodes = sumNode->GetInDataNodes();
  if ((subInputDataNodes.size() != SUB_INPUT_NODE_NUM) || (expNode->GetInDataNodes().size() != EXP_INPUT_NODE_NUM) ||
      (sumInputDataNodes.size() != SUM_INPUT_NODE_NUM) || (sumNode->GetOutDataNodes().size() != SUM_OUTPUT_NODE_NUM) ||
      (mulNode->GetOutDataNodes().size() != MUL_OUTPUT_NODE_NUM)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "this pattern does not meet the fusion condition");
    return FAILED;
  }
  if (subInputDataNodes.at(0) != sumInputDataNodes.at(0)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "sum node and sub node should have the same parent node");
    return FAILED;
  }
  return SUCCESS;
}

Status LogSoftmaxGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr expNode = GetNodeFromMapping(PATTERN_EXP, mapping);
  ge::NodePtr sumNode = GetNodeFromMapping(PATTERN_SUM, mapping);
  FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "mul node is null"), return FAILED);
  FUSION_PASS_CHECK(subNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "sub node is null"), return FAILED);
  FUSION_PASS_CHECK(expNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "exp node is null"), return FAILED);
  FUSION_PASS_CHECK(sumNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "sum node is null"), return FAILED);
  // determine whether the patten node meets the fusion requirements
  if (IsMatch(sumNode, subNode, expNode, mulNode) != SUCCESS) {
    return NOT_CHANGED;
  }
  // fusion step
  if (DoFusion(graph, sumNode, subNode, expNode, mulNode, fusionNodes) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "logSoftMaxGrad fusion failed");
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "LogSoftmaxGrad fusion success!");
  return SUCCESS;
}

Status LogSoftmaxGradFusionPass::DoFusion(ge::ComputeGraph& graph, ge::NodePtr sumNode, ge::NodePtr subNode,
                                          ge::NodePtr expNode, ge::NodePtr mulNode, vector<ge::NodePtr>& fusionNodes) {
  auto logsoftmaxGradOp = ge::OperatorFactory::CreateOperator("logSoftmaxGrad", LOGSOFTMAXGRAD);
  if (logsoftmaxGradOp.IsEmpty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "create fusion node LogSoftmaxGrad op desc error");
    return NOT_CHANGED;
  }
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(logsoftmaxGradOp);
  logsoftmaxGradOp.BreakConnect();

  ge::NodePtr logsoftmaxGradNode = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(logsoftmaxGradNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "logsoftmaxGrad create node failed"),
                    return FAILED);
  fusionNodes.push_back(logsoftmaxGradNode);
  if (UpdateAttr(sumNode, logsoftmaxGradNode) == FAILED) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "attr failed");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::RemoveInputEdge(mulNode) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove mul node input edge failed");
    return FAILED;
  }
  if (PatternFusionUtil::RemoveInputEdge(subNode) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove sub node input edge failed");
    return FAILED;
  }

  auto sumInDataAnchor = sumNode->GetInDataAnchor(0);
  FUSION_PASS_CHECK(sumInDataAnchor == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "sum input data anchor 0 is null"),
                    return FAILED);

  auto preOutDataAnchor = sumInDataAnchor->GetPeerOutAnchor();
  auto firstInDataAnchor = logsoftmaxGradNode->GetInDataAnchor(0);
  FUSION_PASS_CHECK(firstInDataAnchor == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "logSoftMaxGrad input data anchor 0 is null"), return FAILED);

  if (ge::GraphUtils::AddEdge(preOutDataAnchor, firstInDataAnchor) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge failed");
    return FAILED;
  }

  if (PatternFusionUtil::RemoveInputEdge(sumNode) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove sum edge failed");
    return FAILED;
  }

  auto expInDataAnchor = expNode->GetInDataAnchor(0);
  FUSION_PASS_CHECK(expInDataAnchor == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "exp input data anchor 0 is null"),
                    return FAILED);

  auto expPreOutDataAnchor = expInDataAnchor->GetPeerOutAnchor();

  auto secondInDataAnchor = logsoftmaxGradNode->GetInDataAnchor(1);
  FUSION_PASS_CHECK(secondInDataAnchor == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "logSoftMaxGrad input anchor 1 is null"), return FAILED);

  if (ge::GraphUtils::AddEdge(expPreOutDataAnchor, secondInDataAnchor) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge failed");
    return FAILED;
  }

  if (PatternFusionUtil::RemoveInputEdge(expNode) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove exp node edge failed");
    return FAILED;
  }
  // set logsoftmax_grad name
  string subName = subNode->GetOpDesc()->GetName();
  vector<string> nameVec = ge::StringUtils::Split(subName, '/');
  string logSoftmaxGradName = "";
  static int logSoftmaxGradCount = 0;
  for (size_t i = 1; i < nameVec.size(); ++i) {
    logSoftmaxGradName += (nameVec[i - 1] + "/");
  }
  logSoftmaxGradName += LOGSOFTMAXGRAD;

  logsoftmaxGradNode->GetOpDesc()->SetName(logSoftmaxGradName + std::to_string(logSoftmaxGradCount));
  ++logSoftmaxGradCount;
  vector<bool> isInputConst;
  for (auto anchor : logsoftmaxGradNode->GetAllInDataAnchors()) {
    auto peerAnchor = anchor->GetPeerOutAnchor();
    auto node = peerAnchor->GetOwnerNode();
    auto outputTensor = node->GetOpDesc()->GetOutputDesc(peerAnchor->GetIdx());
    FUSION_PASS_CHECK(logsoftmaxGradNode->GetOpDesc()->UpdateInputDesc(anchor->GetIdx(), outputTensor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "update input failed."), return FAILED);

    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node) == "Const") {
      isInputConst.push_back(true);
    } else {
      isInputConst.push_back(false);
    }
  }
  logsoftmaxGradNode->GetOpDesc()->SetIsInputConst(isInputConst);
  // link logsoftmaxGrad node with sub child nodes
  if (LinkOutputEdge(subNode, logsoftmaxGradNode) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "link output edge Failed.");
    return FAILED;
  }

  // remove mul sum and exp node
  if (graph.RemoveNode(mulNode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "mul node remove failed");
    return FAILED;
  }
  if (graph.RemoveNode(expNode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "exp node remove failed");
    return FAILED;
  }
  if (graph.RemoveNode(sumNode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "sum node remove failed");
    return FAILED;
  }
  if (graph.RemoveNode(subNode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "sub node remove failed");
    return FAILED;
  }
  return SUCCESS;
}

Status LogSoftmaxGradFusionPass::UpdateAttr(ge::NodePtr sumNode, ge::NodePtr node) {
  FUSION_PASS_CHECK(sumNode == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "sum node is null"), return FAILED);
  vector<int32_t> axisValue;
  if (ge::AttrUtils::GetListInt(sumNode->GetOpDesc(), ATTR_NAME_CONST, axisValue) == false) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Sum op should have axis attr, but now have no axis attr");
    return FAILED;
  }

  if (ge::AttrUtils::SetListInt(node->GetOpDesc(), AXIS, axisValue) == false) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "set sub axis attribute error");
    return FAILED;
  }
  return SUCCESS;
}

Status LogSoftmaxGradFusionPass::LinkOutputEdge(ge::NodePtr oldNode, ge::NodePtr newNode) {
  ge::OutDataAnchorPtr newOutDataAnchor = newNode->GetOutDataAnchor(0);
  if (newOutDataAnchor == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Parameter[newOutDataAnchor] must not be null.");
    return fe::PARAM_INVALID;
  }
  for (ge::OutDataAnchorPtr anchor : oldNode->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Parameter[anchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    for (ge::InDataAnchorPtr dstAnchor : anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(anchor, dstAnchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(newOutDataAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Replace out data anchor Failed.");
        return FAILED;
      }
      auto peerNode = dstAnchor->GetOwnerNode();
      auto inputTensor = peerNode->GetOpDesc()->GetInputDesc(dstAnchor->GetIdx());
      FUSION_PASS_CHECK(
          newNode->GetOpDesc()->UpdateOutputDesc(newOutDataAnchor->GetIdx(), inputTensor) == ge::GRAPH_FAILED,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "op:%s update output desc failed", newNode->GetName().c_str()), return false);
    }
  }
  ge::OutControlAnchorPtr outControlAnchor = oldNode->GetOutControlAnchor();
  ge::OutControlAnchorPtr newOutControlAnchor = newNode->GetOutControlAnchor();
  if (outControlAnchor != nullptr) {
    for (ge::InControlAnchorPtr dstAnchor : outControlAnchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(outControlAnchor, dstAnchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(newOutControlAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Replace input control anchor Failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
REGISTER_PASS("LogSoftmaxGradFusionPass", BUILT_IN_GRAPH_PASS, LogSoftmaxGradFusionPass);
}  // namespace fe
