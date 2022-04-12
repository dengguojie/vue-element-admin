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
 * \file fusedbatchnorm_fusion_pass.cpp
 * \brief fusedbatchnorm fusion pass
 *   (BatchNorm --> BNTrainingReduce & BNTrainingUpdate)
 */
#include "fusedbatchnorm_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
namespace fe {
vector<FusionPattern*> FusedBatchnormFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status FusedBatchnormFusionPass::Run(ge::ComputeGraph& graph) {
  vector<PassMatchResult> passMatchResultVec;
  fe::Status status = MatchPass(graph, passMatchResultVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to match FusedBatchnormFusionPass.");
  }

  auto matchTimes = static_cast<int32_t>(passMatchResultVec.size());
  int32_t effectTimes = 0;
  if (passMatchResultVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not match with pass FusedBatchnormFusionPass.");
    return SUCCESS;
  }

  // Record anchor map
  for (PassMatchResult& matchResult : passMatchResultVec) {
    RecordOutputAnchorMap(matchResult.batchNormPtr);
    for (auto nodePtr : matchResult.assignSubNodeVec) {
      RecordOutputAnchorMap(nodePtr);
    }
  }

  for (PassMatchResult& matchResult : passMatchResultVec) {
    status = FusionGraphWithPass(graph, matchResult);
    if (status == SUCCESS) {
      effectTimes++;
    }
  }
  // get match times
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(), matchTimes, effectTimes);
  FusionStatisticRecorder::Instance().UpdateGraphFusionMatchTimes(fusionInfo);
  FusionStatisticRecorder::Instance().UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%zu, effectedTimes=%d.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), matchTimes, effectTimes);
  return SUCCESS;
}

Status FusedBatchnormFusionPass::MatchPass(ge::ComputeGraph& graph, vector<PassMatchResult>& passMatchResultVec) {
  vector<NodePtr> batchNormNodeVec;
  Status status = GetAllBatchNormNodes(graph, batchNormNodeVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to get all batchnorm nodes from graph.");
  }
  if (batchNormNodeVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not contain batchnorm node.");
    return SUCCESS;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph has [%d] batchnorm node.", batchNormNodeVec.size());
  for (NodePtr bnNodePtr : batchNormNodeVec) {
    PassMatchResult matchResult;
    if (MatchBatchNormNode(bnNodePtr, matchResult) == SUCCESS) {
      passMatchResultVec.push_back(matchResult);
    }
  }
  return SUCCESS;
}

Status FusedBatchnormFusionPass::GetAllBatchNormNodes(ge::ComputeGraph& graph, vector<NodePtr>& batchNormNodeVec) {
  for (NodePtr node : graph.GetDirectNode()) {
    if (node->GetType() == PASS_OP_TYPE_BATCHNORM) {
      batchNormNodeVec.push_back(node);
    }
  }
  return SUCCESS;
}

Status FusedBatchnormFusionPass::MatchBatchNormNode(NodePtr bnNodePtr, PassMatchResult& matchResult) {
  // BN node has epsilon attr
  OpDescPtr bnOpDescPtr = bnNodePtr->GetOpDesc();
  FUSION_PASS_CHECK(bnOpDescPtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnOpDescPtr is null."),
                    return PARAM_INVALID);
  if (!ge::AttrUtils::HasAttr(bnOpDescPtr, "epsilon")) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The fused batch norm node does not have epsilon attr.");
    return FAILED;
  }
  // bn have 3 input
  int inputDataCount = 0;
  for (InDataAnchorPtr inAnchorPtr : bnNodePtr->GetAllInDataAnchors()) {
    if (inAnchorPtr != nullptr && inAnchorPtr->GetPeerOutAnchor() != nullptr &&
        inAnchorPtr->GetPeerOutAnchor()->GetOwnerNode() != nullptr) {
      NodePtr ownerNode = inAnchorPtr->GetPeerOutAnchor()->GetOwnerNode();
      matchResult.dataNodeVec.push_back(ownerNode);
      matchResult.dataNodeOutAnchorVec.push_back(inAnchorPtr->GetPeerOutAnchor());
      matchResult.removeEdgeVec.push_back({inAnchorPtr, inAnchorPtr->GetPeerOutAnchor()});
      inputDataCount++;
    }
  }
  if (inputDataCount < 3) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The BatchNorm node must have at least 3 valid input.");
    return FAILED;
  }

  // bn have 3 output
  for (NodePtr nodePtr : bnNodePtr->GetOutDataNodes()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Out Node Type is [%s]", nodePtr->GetType().c_str());
  }
  vector<NodePtr> subNodes;
  int outputDataCount = 0;
  for (size_t i = 0; i < bnNodePtr->GetAllOutDataAnchors().size(); i++) {
    OutDataAnchorPtr outAnchorPtr = bnNodePtr->GetAllOutDataAnchors().at(i);
    if (outAnchorPtr != nullptr && outAnchorPtr->GetPeerInDataAnchors().size() > 0) {
      InDataAnchorPtr inDataAnchorPtr = outAnchorPtr->GetPeerInDataAnchors().at(0);
      if (inDataAnchorPtr != nullptr && inDataAnchorPtr->GetOwnerNode() != nullptr) {
        if (i == 0) {
          matchResult.outNodePtr = inDataAnchorPtr->GetOwnerNode();
          OP_LOGI(FUSED_OP_TYPE.c_str(), "outNodePtr Node name is [%s]", matchResult.outNodePtr->GetName().c_str());
        } else if (i == 1 || i == 2) {
          if (inDataAnchorPtr->GetOwnerNode()->GetType() == PASS_OP_TYPE_SUB) {
            subNodes.push_back(inDataAnchorPtr->GetOwnerNode());
            matchResult.removeEdgeVec.push_back({inDataAnchorPtr, outAnchorPtr});
          }
        } else {
          matchResult.bnOtherOutAnchorVec.push_back(outAnchorPtr);
        }
        outputDataCount++;
      }
    }
  }
  if (outputDataCount < 3 || subNodes.size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "The BatchNorm node must have at least 3 output and two of them "
            "should be sub node.");
    return FAILED;
  }

  matchResult.subNodeVec.swap(subNodes);
  matchResult.batchNormPtr = bnNodePtr;

  Status status;
  for (NodePtr subNodePtr : matchResult.subNodeVec) {
    status = MatchSubNode(subNodePtr, matchResult);
    if (status != SUCCESS) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Sub node[%s] does not match.", subNodePtr->GetName().c_str());
      return FAILED;
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "The fused batch norm pass has been matched, the name of batchnorm "
          "node is [%s]",
          bnNodePtr->GetName().c_str());
  return SUCCESS;
}

Status FusedBatchnormFusionPass::MatchSubNode(NodePtr subNodePtr, PassMatchResult& matchResult) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Begin to match sub node[%s].", subNodePtr->GetName().c_str());
  // SubNode has 2 input, the other is variable
  if (subNodePtr->GetAllInDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The sub node does not have 2 input.");
    return FAILED;
  }

  // SubNode has 1 output Mul
  if (subNodePtr->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The sub node has 0 or more than 1 output.");
    return FAILED;
  }

  // find mul node
  NodePtr mulNodePtr = FindOutputNode(subNodePtr, "Mul", matchResult, true);
  if (mulNodePtr == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Sub Node does not have a mul output node.");
    return FAILED;
  }
  matchResult.mulNodeVec.push_back(mulNodePtr);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Find node[%s].", mulNodePtr->GetName().c_str());

  // mul has a const input node
  NodePtr constNode = FindInputNode(mulNodePtr, "Constant", matchResult, true);
  if (constNode == nullptr) {
    constNode = FindInputNode(mulNodePtr, "Const", matchResult, true);
    if (constNode == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Mul Node does not have a const input node.");
        return FAILED;
    }
  }
  if (matchResult.constNodeVec.empty()) {
    matchResult.constNodeVec.push_back(constNode);
  } else {
    if (constNode->GetName() != matchResult.constNodeVec[0]->GetName()) {
      matchResult.constNodeVec.push_back(constNode);
    }
  }

  // mul has a output node assignSub
  if (mulNodePtr->GetOutDataNodesSize() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The mul node has 0 or more than 1 output node.");
    return FAILED;
  }

  // find cast node
  NodePtr castInPtr = nullptr;
  NodePtr castOutPtr = nullptr;
  NodePtr varNodePtr = nullptr;
  NodePtr otherVarNodePtr = nullptr;
  castInPtr = FindOutputNode(mulNodePtr, "Cast", matchResult, true);
  if (castInPtr != nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Find cast node[%s].", castInPtr->GetName().c_str());
    matchResult.castInVec.push_back(castInPtr);

    NodePtr assignSubNodePtr = FindOutputNode(castInPtr, "AssignSub", matchResult, true);
    if (assignSubNodePtr == nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "mul Node does not have a AssignSub output node.");
      return FAILED;
    }
    matchResult.assignSubNodeVec.push_back(assignSubNodePtr);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Find node[%s].", assignSubNodePtr->GetName().c_str());

    // find variable node
    varNodePtr = FindInputNodeByIndex(assignSubNodePtr, 0, matchResult, true);

    // find cast out node
    castOutPtr = FindInputNode(subNodePtr, "Cast", matchResult, true);
    if (castOutPtr == nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Can not find cast out node from sub node.");
      return FAILED;
    }
    matchResult.castOutVec.push_back(castOutPtr);

    otherVarNodePtr = FindInputNodeByIndex(castOutPtr, 0, matchResult, true);
  } else {
    // find Assign Sub node
    NodePtr assignSubNodePtr = FindOutputNode(mulNodePtr, "AssignSub", matchResult, true);
    if (assignSubNodePtr == nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "mul Node does not have a AssignSub output node.");
      return FAILED;
    }

    matchResult.assignSubNodeVec.push_back(assignSubNodePtr);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Find node[%s].", assignSubNodePtr->GetName().c_str());

    // find switch node
    NodePtr switchNodePtr = FindInputNode(subNodePtr, "Switch", matchResult, true);
    if (switchNodePtr != nullptr) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Find switch node[%s].", switchNodePtr->GetName().c_str());
      matchResult.switchNodeVec.push_back(switchNodePtr);
      varNodePtr = FindInputNode(switchNodePtr, "Variable", matchResult, true);
      NodePtr constSwitchNode = FindInputNode(switchNodePtr, "Constant", matchResult, true);
      if (constSwitchNode == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Can not find constant node from switch node.");
        return FAILED;
      }
      NodePtr fwkNodePtr = FindInputNodeByIndex(assignSubNodePtr, 0, matchResult, true);
      if (fwkNodePtr == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Assign Sub Node does not have index 0 input node.");
        return FAILED;
      }
      matchResult.fwkNodeVec.push_back(fwkNodePtr);
      otherVarNodePtr = FindInputNode(fwkNodePtr, "Variable", matchResult, false);
    } else {
      // find variable node
      varNodePtr = FindInputNodeByIndex(subNodePtr, 0, matchResult, true);
      otherVarNodePtr = FindInputNodeByIndex(assignSubNodePtr, 0, matchResult, true);
    }
  }
  if (varNodePtr == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Sub Node does not have index 0 node.");
    return FAILED;
  }
  // variable node has 1 output Assign Sub
  if (varNodePtr->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The variable node has 0 or more than 1 output.");
    return FAILED;
  }
  matchResult.variableNodeVec.push_back(varNodePtr);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Find node[%s].", varNodePtr->GetName().c_str());

  // check varNodePtr and otherVarNodePtr the same node
  if (otherVarNodePtr == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Can not find another variable node.");
    return FAILED;
  }
  if (varNodePtr->GetName() != otherVarNodePtr->GetName()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Sub node and AssignSub node don't link to the same variable node.");
    return FAILED;
  }

  return SUCCESS;
}

Status FusedBatchnormFusionPass::FusionGraphWithPass(ge::ComputeGraph& graph, PassMatchResult& matchResult) {
  string bnStreamLabel = "";
  if (!ge::AttrUtils::GetStr(matchResult.batchNormPtr->GetOpDesc(), STREAM_LABEL, bnStreamLabel)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNorm not have _stream_label attr.");
  }

  for (NodePtr subNodePtr : matchResult.subNodeVec) {
    string streamLabel;
    if (ge::AttrUtils::GetStr(subNodePtr->GetOpDesc(), STREAM_LABEL, streamLabel)) {
      if ((bnStreamLabel.empty() && !streamLabel.empty()) || bnStreamLabel != streamLabel) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
        return NOT_CHANGED;
      }
    } else if (!bnStreamLabel.empty()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
      return NOT_CHANGED;
    }
  }

  for (NodePtr mulNodePtr : matchResult.mulNodeVec) {
    string streamLabel;
    if (ge::AttrUtils::GetStr(mulNodePtr->GetOpDesc(), STREAM_LABEL, streamLabel)) {
      if ((bnStreamLabel.empty() && !streamLabel.empty()) || bnStreamLabel != streamLabel) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
        return NOT_CHANGED;
      }
    } else if (!bnStreamLabel.empty()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
      return NOT_CHANGED;
    }
  }

  for (NodePtr assignSubNodePtr : matchResult.assignSubNodeVec) {
    string streamLabel;
    if (ge::AttrUtils::GetStr(assignSubNodePtr->GetOpDesc(), STREAM_LABEL, streamLabel)) {
      if ((bnStreamLabel.empty() && !streamLabel.empty()) || bnStreamLabel != streamLabel) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
        return NOT_CHANGED;
      }
    } else if (!bnStreamLabel.empty()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
      return NOT_CHANGED;
    }
  }

  NodePtr bnNodePtr = matchResult.batchNormPtr;
  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "Begin to fusion pass, the name of batch norm is [%s].", bnNodePtr->GetName().c_str());
  std::string fusedBatchNormName = bnNodePtr->GetOpDesc()->GetName();
  string op_compile_strategy;
  (void)ge::AttrUtils::GetStr(bnNodePtr->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
  // BNTrainingReduce
  OpDescPtr bnReduceOp;
  FUSION_PASS_MAKE_SHARED(
      bnReduceOp = std::make_shared<ge::OpDesc>(fusedBatchNormName + "_Reduce", PASS_OP_TYPE_BNREDUCE),
      return INTERNAL_ERROR);
  bnReduceOp->AddInputDesc("x", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(0));
  bnReduceOp->AddOutputDesc("sum", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(1));
  bnReduceOp->AddOutputDesc("square_sum", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(2));

  ge::NodePtr bnReduceNode = graph.AddNode(bnReduceOp);
  FUSION_PASS_CHECK(bnReduceNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnReduceNode is null, fusion failed."),
                    return PARAM_INVALID);
  // add edge for bnreduce node
  GraphUtils::AddEdge(matchResult.dataNodeOutAnchorVec[0], bnReduceNode->GetInDataAnchor(0));

  // BNTrainingUpdate
  OpDescPtr bnUpdateOp;
  FUSION_PASS_MAKE_SHARED(
      bnUpdateOp = std::make_shared<ge::OpDesc>(fusedBatchNormName + "_Update", PASS_OP_TYPE_BNUPDATE),
      return INTERNAL_ERROR);
  bnUpdateOp->AddInputDesc("x", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(0));
  bnUpdateOp->AddInputDesc("sum", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(1));
  bnUpdateOp->AddInputDesc("square_sum", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(2));
  bnUpdateOp->AddInputDesc("scale", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(1));
  bnUpdateOp->AddInputDesc("offset", matchResult.batchNormPtr->GetOpDesc()->GetInputDesc(2));
  bnUpdateOp->AddInputDesc("mean", matchResult.variableNodeVec[0]->GetOpDesc()->GetOutputDesc(0));
  bnUpdateOp->AddInputDesc("variance", matchResult.variableNodeVec[1]->GetOpDesc()->GetOutputDesc(0));

  bnUpdateOp->AddOutputDesc("y", bnNodePtr->GetOpDesc()->GetOutputDesc(0));
  bnUpdateOp->AddOutputDesc("mean", matchResult.variableNodeVec[0]->GetOpDesc()->GetOutputDesc(0));
  bnUpdateOp->AddOutputDesc("variance", matchResult.variableNodeVec[1]->GetOpDesc()->GetOutputDesc(0));
  bnUpdateOp->AddOutputDesc("batch_mean", bnNodePtr->GetOpDesc()->GetOutputDesc(1));
  bnUpdateOp->AddOutputDesc("batch_variance", bnNodePtr->GetOpDesc()->GetOutputDesc(2));

  ge::NodePtr bnUpdateNode = graph.AddNode(bnUpdateOp);

  FUSION_PASS_CHECK(bnUpdateNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnUpdateNode is null, fusion failed."),
                    return PARAM_INVALID);

  // add edge for input of bnupdate
  GraphUtils::AddEdge(matchResult.dataNodeOutAnchorVec[0], bnUpdateNode->GetInDataAnchor(0));
  GraphUtils::AddEdge(bnReduceNode->GetOutDataAnchor(0), bnUpdateNode->GetInDataAnchor(1));
  GraphUtils::AddEdge(bnReduceNode->GetOutDataAnchor(1), bnUpdateNode->GetInDataAnchor(2));
  GraphUtils::AddEdge(matchResult.dataNodeOutAnchorVec[1], bnUpdateNode->GetInDataAnchor(3));
  GraphUtils::AddEdge(matchResult.dataNodeOutAnchorVec[2], bnUpdateNode->GetInDataAnchor(4));
  if (matchResult.fwkNodeVec.size() == 2) {
    GraphUtils::AddEdge(matchResult.assignSubNodeVec[0]->GetInDataAnchor(0)->GetPeerOutAnchor(),
                        bnUpdateNode->GetInDataAnchor(5));
    GraphUtils::AddEdge(matchResult.assignSubNodeVec[1]->GetInDataAnchor(0)->GetPeerOutAnchor(),
                        bnUpdateNode->GetInDataAnchor(6));
  } else {
    GraphUtils::AddEdge(matchResult.variableNodeVec[0]->GetOutDataAnchor(0), bnUpdateNode->GetInDataAnchor(5));
    GraphUtils::AddEdge(matchResult.variableNodeVec[1]->GetOutDataAnchor(0), bnUpdateNode->GetInDataAnchor(6));
  }

  // add edge for output of bnupdate
  OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of netoutput is [%d].",
          matchResult.outNodePtr->GetAllInDataAnchors().size());

  for (InDataAnchorPtr inAnchorPtr : matchResult.batchNormPtr->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(0), inAnchorPtr);
  }

  if (matchResult.assignSubNodeVec[0]->GetAllOutDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of inAnFromAssignSubZero is [%d].",
            matchResult.assignSubNodeVec[0]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    OutDataAnchorPtr assignSubOutAnchorPtr = matchResult.assignSubNodeVec[0]->GetOutDataAnchor(0);
    if (assignSubOutAnchorPtr != nullptr && assignSubOutAnchorPtr->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr inAnchorPtr : assignSubOutAnchorPtr->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(1), inAnchorPtr);
      }
    }
  }
  if (matchResult.assignSubNodeVec[1]->GetAllOutDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of inAnFromAssignSubOne is [%d].",
            matchResult.assignSubNodeVec[1]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    OutDataAnchorPtr assignSubOutAnchorPtr = matchResult.assignSubNodeVec[1]->GetOutDataAnchor(0);
    if (assignSubOutAnchorPtr != nullptr && assignSubOutAnchorPtr->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr inAnchorPtr : assignSubOutAnchorPtr->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(2), inAnchorPtr);
      }
    }
  }
  // remove control anchor of assign sub
  for (NodePtr assignSubNodePtr : matchResult.assignSubNodeVec) {
    if (assignSubNodePtr->GetOutControlAnchor() != nullptr) {
      OutControlAnchorPtr assignSubOutCtrlAnchorPtr = assignSubNodePtr->GetOutControlAnchor();
      if (assignSubOutCtrlAnchorPtr->GetPeerInControlAnchors().size() > 0) {
        for (InControlAnchorPtr inCtrlAnchorPtr : assignSubOutCtrlAnchorPtr->GetPeerInControlAnchors()) {
          GraphUtils::RemoveEdge(assignSubOutCtrlAnchorPtr, inCtrlAnchorPtr);
        }
      }
    }
  }
  if (!matchResult.bnOtherOutAnchorVec.empty()) {
    for (InDataAnchorPtr otherInAnchor : matchResult.bnOtherOutAnchorVec[0]->GetPeerInDataAnchors()) {
      otherInAnchor->UnlinkAll();
      GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(3), otherInAnchor);
    }
    if (matchResult.bnOtherOutAnchorVec.size() == 2) {  // tf1.12 bn has 5 output
      for (InDataAnchorPtr otherInAnchor : matchResult.bnOtherOutAnchorVec[1]->GetPeerInDataAnchors()) {
        otherInAnchor->UnlinkAll();
        GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(4), otherInAnchor);
      }
    } else if (matchResult.bnOtherOutAnchorVec.size() == 3) {  // tf1.15 bn has 6 output
      for (InDataAnchorPtr otherInAnchor : matchResult.bnOtherOutAnchorVec[1]->GetPeerInDataAnchors()) {
        otherInAnchor->UnlinkAll();
        GraphUtils::AddEdge(bnUpdateNode->GetOutDataAnchor(4), otherInAnchor);
      }
      // delete the sixth output
      for (InDataAnchorPtr otherInAnchor : matchResult.bnOtherOutAnchorVec[2]->GetPeerInDataAnchors()) {
        auto bnOtherNodePtr = otherInAnchor->GetOwnerNode();
        int otherIdx = otherInAnchor->GetIdx();
        otherInAnchor->UnlinkAll();
        ge::NodeUtils::ClearInDataAnchor(bnOtherNodePtr, otherInAnchor);
        ge::OpDescUtils::ClearInputDesc(bnOtherNodePtr->GetOpDesc(), (uint32_t)otherIdx);
      }
    } else if (matchResult.bnOtherOutAnchorVec.size() > 3) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "BN op has more than 6 output, check its rationality.");
      return NOT_CHANGED;
    }
  }

  // Add attr
  float epsilonVal;
  if (ge::AttrUtils::GetFloat(bnNodePtr->GetOpDesc(), "epsilon", epsilonVal)) {
    if (!ge::AttrUtils::SetFloat(bnUpdateNode->GetOpDesc(), "epsilon", epsilonVal)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to set epsilon for bn update node.");
      return FAILED;
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to get epsilon from batchnorm node.");
    return FAILED;
  }

  if (!bnStreamLabel.empty()) {
    if (!ge::AttrUtils::SetStr(bnUpdateNode->GetOpDesc(), STREAM_LABEL, bnStreamLabel)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnUpdateNode set _stream_label error, fusion failed.");
      return FAILED;
    }
    if (!ge::AttrUtils::SetStr(bnReduceNode->GetOpDesc(), STREAM_LABEL, bnStreamLabel)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnReduceNode set _stream_label error, fusion failed.");
      return FAILED;
    }
  }

  InheritAttrFromOriginNodes(bnUpdateNode->GetOpDesc(), bnReduceNode->GetOpDesc(), op_compile_strategy);

  Operator mulOp = ge::OpDescUtils::CreateOperatorFromNode(matchResult.mulNodeVec[0]);
  int constInputIndex = 1;
  for (InDataAnchorPtr muInpDataAnchor : matchResult.mulNodeVec[0]->GetAllInDataAnchors()) {
    if (muInpDataAnchor == nullptr) {
      continue;
    }
    OutDataAnchorPtr mulOutDataAnchorPtr = muInpDataAnchor->GetPeerOutAnchor();
    NodePtr mulInNode = mulOutDataAnchorPtr->GetOwnerNode();
    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(mulInNode) == "Constant") {
      constInputIndex = muInpDataAnchor->GetIdx();
      break;
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "The index of mul node's const input node is %d", constInputIndex);
  Tensor constTensor;
  mulOp.GetInputConstData(matchResult.mulNodeVec[0]->GetOpDesc()->GetInputNameByIndex(constInputIndex).c_str(),
                          constTensor);
  DataType constDataType = matchResult.mulNodeVec[0]->GetOpDesc()->GetInputDesc(constInputIndex).GetDataType();
  if (constTensor.GetData() != nullptr) {
    if (constDataType == ge::DT_FLOAT16) {
      uint16_t* constDataPtr = reinterpret_cast<uint16_t*>(constTensor.GetData());
      uint16_t constData = static_cast<uint16_t>(*constDataPtr);
      fp16_t constDataFp16(constData);
      float constDataFp32 = constDataFp16.toFloat();
      OP_LOGI(FUSED_OP_TYPE.c_str(), "factor is %f", constDataFp32);
      if (!AttrUtils::SetFloat(bnUpdateNode->GetOpDesc(), "factor", constDataFp32)) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to set attr factor for bn update node.");
    return FAILED;
      }
    } else {
      float* constDataPtr = reinterpret_cast<float*>(constTensor.GetData());
      float constData = static_cast<float>(*constDataPtr);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "factor is %f", constData);
      if (!AttrUtils::SetFloat(bnUpdateNode->GetOpDesc(), "factor", constData)) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to set attr factor for bn update node.");
    return FAILED;
      }
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The const value is nullptr.");
    return FAILED;
  }

  // remove edges
  for (PassRemoveEdge passRemoveEdge : matchResult.removeEdgeVec) {
    GraphUtils::RemoveEdge(passRemoveEdge.inAnchorPtr, passRemoveEdge.outAnchorPtr);
  }
  std::vector<ge::NodePtr> originalNodes;
  for (NodePtr asgSub : matchResult.assignSubNodeVec) {
    for (OutDataAnchorPtr outAnchor : asgSub->GetAllOutDataAnchors()) {
      outAnchor->UnlinkAll();
    }
    graph.RemoveNode(asgSub);
    originalNodes.push_back(asgSub);
  }
  for (NodePtr mul : matchResult.mulNodeVec) {
    graph.RemoveNode(mul);
    originalNodes.push_back(mul);
  }
  for (NodePtr sub : matchResult.subNodeVec) {
    graph.RemoveNode(sub);
    originalNodes.push_back(sub);
  }
  if (!matchResult.switchNodeVec.empty()) {
    for (NodePtr swt : matchResult.switchNodeVec) {
      graph.RemoveNode(swt);
      originalNodes.push_back(swt);
    }
  }
  if (!matchResult.castInVec.empty()) {
    for (NodePtr castin : matchResult.castInVec) {
      graph.RemoveNode(castin);
      originalNodes.push_back(castin);
    }
  }
  if (!matchResult.castOutVec.empty()) {
    for (NodePtr castout : matchResult.castOutVec) {
      graph.RemoveNode(castout);
      originalNodes.push_back(castout);
    }
  }

  graph.RemoveNode(matchResult.batchNormPtr);
  originalNodes.push_back(matchResult.batchNormPtr);

  for (NodePtr constNodePtr : matchResult.constNodeVec) {
    if (PatternFusionUtil::GetOutEdgeSize(constNodePtr) == 0) {
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNodePtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Remove Node[%s] failed", constNodePtr->GetName().c_str()),
                        return FAILED);
      originalNodes.push_back(constNodePtr);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNodePtr->GetName().c_str());
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.", constNodePtr->GetName().c_str());
    }
  }

  GraphPassUtil::RecordOriginalNames(originalNodes, bnReduceNode);
  GraphPassUtil::RecordOriginalNames(originalNodes, bnUpdateNode);
  ge::AttrUtils::SetBool(bnReduceOp, ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, true);
  ge::AttrUtils::SetBool(bnUpdateOp, ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, true);
  FUSION_PASS_CHECK(matchResult.batchNormPtr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "BatchNormPtr is null"),
                    return FAILED);
  FUSION_PASS_CHECK(matchResult.batchNormPtr->GetOpDesc() == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "BatchNorm OpDesc is null"), return FAILED);
  if (matchResult.batchNormPtr->GetOpDesc()->GetOutputsSize() >= 1) {
    SetOutputTensorDescAttr(0, 0, matchResult.batchNormPtr, bnUpdateNode);
  }
  if (matchResult.batchNormPtr->GetOpDesc()->GetOutputsSize() >= 5) {
    SetOutputTensorDescAttr(3, 3, matchResult.batchNormPtr, bnUpdateNode);
    SetOutputTensorDescAttr(4, 4, matchResult.batchNormPtr, bnUpdateNode);
  }
  if (matchResult.assignSubNodeVec.size() == 2) {
    SetOutputTensorDescAttr(0, 1, matchResult.assignSubNodeVec[0], bnUpdateNode);
    SetOutputTensorDescAttr(0, 2, matchResult.assignSubNodeVec[1], bnUpdateNode);
  }

  std::ostringstream reduceOSS;
  reduceOSS << bnReduceOp->GetName() << " ";
  for (GeTensorDesc& tensorDesc : bnReduceOp->GetAllInputsDesc()) {
    reduceOSS << endl << "Format:" << tensorDesc.GetFormat() << " OriFormat" << tensorDesc.GetOriginFormat();
    reduceOSS << " Dim:{";
    for (int64_t dim : tensorDesc.GetShape().GetDims()) {
      reduceOSS << dim << " ";
    }
    reduceOSS << "}";
    reduceOSS << " Origin Dim:{";
    for (int64_t dim : tensorDesc.GetOriginShape().GetDims()) {
      reduceOSS << dim << " ";
    }
    reduceOSS << "}";
  }
  for (GeTensorDesc& tensorDesc : bnReduceOp->GetAllOutputsDesc()) {
    reduceOSS << endl << "Format:" << tensorDesc.GetFormat() << " OriFormat" << tensorDesc.GetOriginFormat();
    reduceOSS << " Dim:{";
    for (int64_t dim : tensorDesc.GetShape().GetDims()) {
      reduceOSS << dim << " ";
    }
    reduceOSS << "}";
    reduceOSS << " Origin Dim:{";
    for (int64_t dim : tensorDesc.GetOriginShape().GetDims()) {
      reduceOSS << dim << " ";
    }
    reduceOSS << "}";
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BNReduce : %s", reduceOSS.str().c_str());

  std::ostringstream updateOSS;
  updateOSS << bnUpdateOp->GetName() << " ";
  for (GeTensorDesc& tensorDesc : bnUpdateOp->GetAllInputsDesc()) {
    updateOSS << "Format:" << tensorDesc.GetFormat() << " OriFormat" << tensorDesc.GetOriginFormat();
    updateOSS << " Dim:{";
    for (int64_t dim : tensorDesc.GetShape().GetDims()) {
      updateOSS << dim << " ";
    }
    updateOSS << "}";
    updateOSS << " Origin Dim:{";
    for (int64_t dim : tensorDesc.GetOriginShape().GetDims()) {
      updateOSS << dim << " ";
    }
    updateOSS << "}";
  }
  for (GeTensorDesc& tensorDesc : bnUpdateOp->GetAllOutputsDesc()) {
    updateOSS << "Format:" << tensorDesc.GetFormat() << " OriFormat" << tensorDesc.GetOriginFormat();
    updateOSS << " Dim:{";
    for (int64_t dim : tensorDesc.GetShape().GetDims()) {
      updateOSS << dim << " ";
    }
    updateOSS << "}";
    updateOSS << " Origin Dim:{";
    for (int64_t dim : tensorDesc.GetOriginShape().GetDims()) {
      updateOSS << dim << " ";
    }
    updateOSS << "}";
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "BNUpdate : %s", updateOSS.str().c_str());

  return SUCCESS;
}

void FusedBatchnormFusionPass::InheritAttrFromOriginNodes(const ge::OpDescPtr &bn_update_op,
                                                          const ge::OpDescPtr &bn_reduce_op,
                                                          const string &op_compile_strategy) {
  if (!op_compile_strategy.empty()) {
    (void)ge::AttrUtils::SetStr(bn_update_op, ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
    (void)ge::AttrUtils::SetStr(bn_reduce_op, ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
  }
}

NodePtr FusedBatchnormFusionPass::FindInputNode(NodePtr nodePtr, string opType, PassMatchResult& matchResult,
                                                bool isRemoveEdge) {
  if (nodePtr->GetAllInDataAnchors().empty()) {
    return nullptr;
  }
  for (InDataAnchorPtr inDataAnchorPtr : nodePtr->GetAllInDataAnchors()) {
    if (inDataAnchorPtr == nullptr) {
      return nullptr;
    }
    OutDataAnchorPtr outDataAnchorPtr = inDataAnchorPtr->GetPeerOutAnchor();
    if (inDataAnchorPtr == nullptr) {
      return nullptr;
    }
    NodePtr ownerNode = outDataAnchorPtr->GetOwnerNode();
    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(ownerNode) == opType) {
      if (isRemoveEdge) {
        matchResult.removeEdgeVec.push_back({inDataAnchorPtr, outDataAnchorPtr});
      }
      return ownerNode;
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Do not find the input node.");
  return nullptr;
}

NodePtr FusedBatchnormFusionPass::FindInputNodeByIndex(NodePtr nodePtr, unsigned int index,
                                                       PassMatchResult& matchResult, bool isRemoveEdge) {
  if (nodePtr->GetAllInDataAnchors().empty()) {
    return nullptr;
  }
  if (nodePtr->GetAllInDataAnchors().size() <= index) {
    return nullptr;
  }
  InDataAnchorPtr inDataAnchorPtr = nodePtr->GetInDataAnchor(index);
  if (inDataAnchorPtr == nullptr) {
    return nullptr;
  }
  OutDataAnchorPtr outDataAnchorPtr = inDataAnchorPtr->GetPeerOutAnchor();
  if (inDataAnchorPtr == nullptr) {
    return nullptr;
  }
  NodePtr ownerNode = outDataAnchorPtr->GetOwnerNode();
  if (ownerNode == nullptr) {
    return nullptr;
  }
  if (isRemoveEdge) {
    matchResult.removeEdgeVec.push_back({inDataAnchorPtr, outDataAnchorPtr});
  }
  return ownerNode;
}

NodePtr FusedBatchnormFusionPass::FindOutputNode(NodePtr nodePtr, string opType, PassMatchResult& matchResult,
                                                 bool isRemoveEdge) {
  if (nodePtr->GetAllOutDataAnchors().empty()) {
    return nullptr;
  }
  for (OutDataAnchorPtr outDataAnchor : nodePtr->GetAllOutDataAnchors()) {
    if (outDataAnchor == nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", nodePtr->GetName().c_str());
      return nullptr;
    }
    if (outDataAnchor->GetPeerInDataAnchors().empty()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", nodePtr->GetName().c_str());
      return nullptr;
    }
    for (InDataAnchorPtr inDataAnchorPtr : outDataAnchor->GetPeerInDataAnchors()) {
      if (inDataAnchorPtr == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr in Data anchor.", nodePtr->GetName().c_str());
        return nullptr;
      }
      NodePtr ownerNode = inDataAnchorPtr->GetOwnerNode();
      if (ownerNode->GetType() == opType) {
        if (isRemoveEdge) {
          matchResult.removeEdgeVec.push_back({inDataAnchorPtr, outDataAnchor});
        }
        return ownerNode;
      }
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Do not find the output node.");
  return nullptr;
}

NodePtr FusedBatchnormFusionPass::FindOutputNodeByName(NodePtr nodePtr, string opName, PassMatchResult& matchResult,
                                                       bool isRemoveEdge) {
  if (nodePtr->GetAllOutDataAnchors().empty()) {
    return nullptr;
  }
  for (OutDataAnchorPtr outDataAnchor : nodePtr->GetAllOutDataAnchors()) {
    if (outDataAnchor == nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", nodePtr->GetName().c_str());
      return nullptr;
    }
    if (outDataAnchor->GetPeerInDataAnchors().empty()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", nodePtr->GetName().c_str());
      return nullptr;
    }
    for (InDataAnchorPtr inDataAnchorPtr : outDataAnchor->GetPeerInDataAnchors()) {
      if (inDataAnchorPtr == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr in Data anchor.", nodePtr->GetName().c_str());
        return nullptr;
      }
      NodePtr ownerNode = inDataAnchorPtr->GetOwnerNode();
      if (ownerNode->GetName() == opName) {
        if (isRemoveEdge) {
          matchResult.removeEdgeVec.push_back({inDataAnchorPtr, outDataAnchor});
        }
        return ownerNode;
      }
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Do not find the output node.");
  return nullptr;
}

Status FusedBatchnormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  return SUCCESS;
}

Status FusedBatchnormFusionPass::SetOutputTensorDescAttr(uint16_t originOutputIndex, uint16_t fuseOutputIndex,
                                                         ge::NodePtr originNode, ge::NodePtr fuseNode) {
  FUSION_PASS_CHECK(fuseNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode is null"), return FAILED);
  FUSION_PASS_CHECK(originNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "originNode is null"), return FAILED);
  FUSION_PASS_CHECK(fuseNode->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode OpDesc is null"),
                    return FAILED);
  FUSION_PASS_CHECK(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex) == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode outputDesc is null"), return FAILED);
  FUSION_PASS_CHECK(originNode->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "originNode OpDesc is null"),
                    return FAILED);
  ge::AttrUtils::SetStr(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME,
                        originNode->GetName());
  ge::AttrUtils::SetInt(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex),
                        ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, originOutputIndex);
  GraphPassUtil::SetDataDumpOriginDataType(
      originNode->GetOpDesc()->GetOutputDesc(originOutputIndex).GetOriginDataType(),
      fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex));
  GraphPassUtil::SetDataDumpOriginFormat(originNode->GetOpDesc()->GetOutputDesc(originOutputIndex).GetOriginFormat(),
                                         fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex));
  return SUCCESS;
}

REGISTER_PASS("FusedBatchnormFusionPass", BUILT_IN_GRAPH_PASS, FusedBatchnormFusionPass);
}  // namespace fe
