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
 * \file remove_node_fusion_pass.cpp
 * \brief remove useless node framework code
 */
#include "remove_node_fusion_pass.h"
#include "fusion_precheck_func.h"
#include "graph/utils/graph_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
REGISTER_PASS("ARemoveNodeFusionPass", BUILT_IN_GRAPH_PASS, RemoveNodeFusionPass);

static const uint32_t PASS_UINT_ONE = 1;
static const std::string kOpTypeData = "Data";

vector<FusionPattern*> RemoveNodeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status RemoveNodeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& newNodes) {
  return SUCCESS;
}

Status RemoveNodeFusionPass::Run(ge::ComputeGraph& graph) {
  map<NodePtr, vector<LinkIndexPair>> removeNodeMap;
  for (NodePtr node : graph.GetDirectNode()) {
    string opType = node->GetOpDesc()->GetType();

    // get register info by op type
    FusionRemoveNodeRegister reg;
    Status status = FusionRemoveNodeRegistry::Instance()->GetRegisterByOpType(opType, reg);
    if (status != SUCCESS) {
      continue;
    }

    vector<LinkIndexPair> linkIndexPairVec;
    function<Status(NodePtr)> preCheckFunc;
    reg.GetParameters(linkIndexPairVec, preCheckFunc);

    // invoke check function if there is one.
    if (preCheckFunc != nullptr) {
        Status preCheckFuncResult = preCheckFunc(node);
        if (preCheckFuncResult != SUCCESS && preCheckFuncResult != NOT_CHANGED) {
            OP_LOGD(opType.c_str(), "Remove node pre-check of node[%s, %s] is not passed, return failed.",
                    node->GetName().c_str(), opType.c_str());
            return FAILED;
        }

        if (preCheckFuncResult == NOT_CHANGED) {
            OP_LOGD(opType.c_str(), "Remove node pre-check of node[%s, %s] is not passed, not changed.",
                    node->GetName().c_str(), opType.c_str());
            continue;
        }
    }
    removeNodeMap.emplace(node, linkIndexPairVec);
  }

  if (removeNodeMap.empty()) {
    OP_LOGI("RemoveNodeFusion", "There is no node needs to be removed.");
    return SUCCESS;
  }

  auto matchTimes = static_cast<int32_t>(removeNodeMap.size());
  int32_t effectTimes = 0;
  // iterate removeNodeMap. handle every node
  for (auto& item : removeNodeMap) {
    Status status = RemoveNode(item.first, item.second, graph);
    FUSION_PASS_CHECK(status != SUCCESS && status != NOT_CHANGED,
                      VECTOR_FUSION_INNER_ERR_REPORT(item.first->GetType().c_str(), "Fail to remove node[%s] in remove node fusion pass.",
                              item.first->GetName().c_str()),
                      return FAILED);
    if (status == SUCCESS) {
      effectTimes++;
    }
  }
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(), matchTimes, effectTimes);
  FusionStatisticRecorder::Instance().UpdateGraphFusionMatchTimes(fusionInfo);
  FusionStatisticRecorder::Instance().UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD("ARemoveNodeFusionPass",
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%zu, effectedTimes=%d.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), matchTimes, effectTimes);
  return SUCCESS;
}

Status RemoveNodeFusionPass::RemoveNode(NodePtr node, vector<LinkIndexPair> linkIndexPairVec, ge::ComputeGraph& graph) {
  string nodeType = node->GetType();
  string nodeName = node->GetName();
  OP_LOGD(nodeType.c_str(), "Begin to remove node[%s].", nodeName.c_str());
  // get node linked inAnchor and outAnchor
  map<uint32_t, InDataAnchorPtr> inDataAnchorMap;
  map<uint32_t, OutDataAnchorPtr> outDataAnchorMap;
  map<uint32_t, OutDataAnchorPtr> inputOutDataAnchorMap;
  map<uint32_t, vector<InDataAnchorPtr>> outputInDataAnchorMap;
  GetLinkedDataAnchor(node, inDataAnchorMap, outDataAnchorMap, inputOutDataAnchorMap, outputInDataAnchorMap);

  // if linked inAnchor or outAnchor is empty, do nothing
  FUSION_PASS_CHECK(
      inDataAnchorMap.empty() || outDataAnchorMap.empty(),
      OP_LOGD(nodeType.c_str(), "Node[%s] does not link to any input nodes or output nodes.", nodeName.c_str()),
      return NOT_CHANGED);

  // verify link pair. make sure every outDataAnchor have only one inDataAnchor
  FUSION_PASS_CHECK(!VerifyLinkPair(inDataAnchorMap, outDataAnchorMap, linkIndexPairVec),
                    OP_LOGD(nodeType.c_str(), "Fail to verify the link pair, this node remains the same."),
                    return NOT_CHANGED);

  // verify and handle unlinked in data anchor
  Status status = HandleUnlinkedInAnchor(inDataAnchorMap, linkIndexPairVec, graph);
  FUSION_PASS_CHECK(status == NOT_CHANGED,
                    OP_LOGD(nodeType.c_str(),
                            "This node[%s] can not be removed "
                            "for some in data anchors can not be removed.",
                            nodeName.c_str()),
                    return status);
  FUSION_PASS_CHECK(
      status != NOT_CHANGED && status != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(nodeType.c_str(), "Fail to handle the unlinked in data anchor of node[%s].", nodeName.c_str()),
      return status);

  // relink control anchor
  status = ReLinkControlAnchor(node);
  FUSION_PASS_CHECK(
      status != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(nodeType.c_str(), "Fail to re-link ctrl edge of node[%s] while removing it.", nodeName.c_str()),
      return FAILED);

  // relink data anchor
  status = ReLinkDataAnchor(linkIndexPairVec, inDataAnchorMap, outDataAnchorMap, inputOutDataAnchorMap,
                            outputInDataAnchorMap);
  FUSION_PASS_CHECK(
      status != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(nodeType.c_str(), "Fail to re-link data edge of node[%s] while removing it.", nodeName.c_str()),
      return FAILED);

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    VECTOR_FUSION_INNER_ERR_REPORT(nodeType.c_str(), "Fail to remove Node[%s].", nodeName.c_str()), return FAILED);

  OP_LOGI(nodeType.c_str(), "The node[%s, %s] has been removed successfully.", nodeName.c_str(),
          node->GetType().c_str());
  return SUCCESS;
}

void RemoveNodeFusionPass::GetLinkedDataAnchor(NodePtr node, map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                                               map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
                                               map<uint32_t, OutDataAnchorPtr>& inputOutDataAnchorMap,
                                               map<uint32_t, vector<InDataAnchorPtr>>& outputInDataAnchorMap) {
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); i++) {
    InDataAnchorPtr inDataAnchor = node->GetInDataAnchor(i);
    if (inDataAnchor == nullptr) {
      continue;
    }
    if (inDataAnchor->GetPeerOutAnchor() != nullptr) {
      inDataAnchorMap.emplace(i, inDataAnchor);
      inputOutDataAnchorMap.emplace(i, inDataAnchor->GetPeerOutAnchor());
    }
  }

  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    OutDataAnchorPtr outDataAnchor = node->GetOutDataAnchor(i);
    if (outDataAnchor == nullptr) {
      continue;
    }
    if (!outDataAnchor->GetPeerInDataAnchors().empty()) {
      outDataAnchorMap.emplace(i, outDataAnchor);
      vector<InDataAnchorPtr> inDataAnchorVec;
      for (InDataAnchorPtr inAnchor : outDataAnchor->GetPeerInDataAnchors()) {
        inDataAnchorVec.push_back(inAnchor);
      }
      outputInDataAnchorMap.emplace(i, inDataAnchorVec);
    }
  }
}

bool RemoveNodeFusionPass::VerifyLinkPair(map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                                          map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
                                          vector<LinkIndexPair>& linkIndexPairVec) {
  vector<LinkIndexPair> newLinkIndexPairVec;
  // if this node has only one input, there is no need to verify the link pair
  // just link the only inDataAnchor to all the outDataAnchors
  if (inDataAnchorMap.size() == PASS_UINT_ONE) {
    OP_LOGD("VerifyLinkPair",
            "This node has only one input node,"
            " this only input node shall be linked to all the output nodes"
            " after this node has been removed.");
    uint32_t inputDataAnchorIndex = inDataAnchorMap.begin()->first;
    for (auto item : outDataAnchorMap) {
      newLinkIndexPairVec.push_back({inputDataAnchorIndex, item.first});
      OP_LOGD("VerifyLinkPair", "Valid link pair [%u, %u].", inputDataAnchorIndex, item.first);
    }
    linkIndexPairVec = newLinkIndexPairVec;
    return true;
  }
  // filter valid link pair
  // check every out anchor has only one inanchor
  map<uint32_t, InDataAnchorPtr>::iterator inAnchorIter;
  map<uint32_t, OutDataAnchorPtr>::iterator outAnchorIter;
  map<uint32_t, uint32_t> outIndexMap;
  map<uint32_t, uint32_t>::iterator outIndexIter;
  for (LinkIndexPair& linkPair : linkIndexPairVec) {
    inAnchorIter = inDataAnchorMap.find(linkPair.inAnchorIndex);
    outAnchorIter = outDataAnchorMap.find(linkPair.outAnchorIndex);
    if (inAnchorIter == inDataAnchorMap.end() || outAnchorIter == outDataAnchorMap.end()) {
      continue;
    }

    outIndexIter = outIndexMap.find(linkPair.outAnchorIndex);
    if (outIndexIter == outIndexMap.end()) {
      OP_LOGD("VerifyLinkPair", "Valid link pair [%u, %u].", linkPair.inAnchorIndex,
              linkPair.outAnchorIndex);  // valid link pair
      newLinkIndexPairVec.push_back(linkPair);
      outIndexMap.emplace(linkPair.outAnchorIndex, linkPair.inAnchorIndex);
    } else if (outIndexIter->second != linkPair.inAnchorIndex) {
      OP_LOGD("VerifyLinkPair", "The out anchor index[%u] has more than one paired in anchor.",
              linkPair.outAnchorIndex);
      return false;
    }
  }
  FUSION_PASS_CHECK(newLinkIndexPairVec.empty(),
                    OP_LOGD("VerifyLinkPair", "There is no valid link index pair according to node."), return false);
  // update link index pair
  linkIndexPairVec = newLinkIndexPairVec;

  // check every out data anchor has inanchor
  for (outAnchorIter = outDataAnchorMap.begin(); outAnchorIter != outDataAnchorMap.end(); outAnchorIter++) {
    if (outIndexMap.find(outAnchorIter->first) == outIndexMap.end()) {
      OP_LOGD("VerifyLinkPair", "The out anchor index[%u] does not have any paired inDataAnchor.",
              outAnchorIter->first);
      return false;
    }
  }
  return true;
}

Status RemoveNodeFusionPass::HandleUnlinkedInAnchor(map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                                                    vector<LinkIndexPair>& linkIndexPairVec, ge::ComputeGraph& graph) {
  // if there is only one input data anchor,
  // this input anchor must be used to link to the output and can not be removed
  if (inDataAnchorMap.size() == PASS_UINT_ONE) {
    return SUCCESS;
  }
  // find unlinked inDataAnchor
  vector<InDataAnchorPtr> unlinkedInAnchorVec;
  map<uint32_t, InDataAnchorPtr>::iterator inAnchorIter;
  for (inAnchorIter = inDataAnchorMap.begin(); inAnchorIter != inDataAnchorMap.end(); inAnchorIter++) {
    bool isLink = false;
    for (LinkIndexPair& linkPair : linkIndexPairVec) {
      if (linkPair.inAnchorIndex == inAnchorIter->first) {
        isLink = true;
        break;
      }
    }
    if (!isLink) {
      unlinkedInAnchorVec.push_back(inAnchorIter->second);
    }
  }
  if (unlinkedInAnchorVec.empty()) {
    return SUCCESS;
  }

  vector<InAnchorRelatedInfo> relatedInfoVec;
  for (InDataAnchorPtr inDataAnchor : unlinkedInAnchorVec) {
    InAnchorRelatedInfo relatedInfo;
    bool canBeRemoved = VerifyRemoveInAnchor(inDataAnchor, relatedInfo);
    if (canBeRemoved) {
      relatedInfoVec.push_back(relatedInfo);
    } else {
      return NOT_CHANGED;
    }
  }

  // remove in data anchor
  for (InAnchorRelatedInfo inAnchorRelatedInfo : relatedInfoVec) {
    for (NodePtr removeNode : inAnchorRelatedInfo.removeNode) {
      // re-link control anchor of removing node
      FUSION_PASS_CHECK(ReLinkControlAnchor(removeNode) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT("HandleUnlinkedInAnchor",
                                "Fail to re-link control edges before removing Node[%s]"
                                " while handling unlinked inDataAnchor.",
                                removeNode->GetName().c_str()),
                        return FAILED);
    }

    // remove edge first
    for (RemoveEdgePair removeEdgePair : inAnchorRelatedInfo.removeEdgeVec) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(removeEdgePair.inAnchorPtr, removeEdgePair.outAnchorPtr) != ge::GRAPH_SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT("HandleUnlinkedInAnchor", "Fail to remove edge while handling unlinked inDataAnchor."),
          return FAILED);
    }

    // then remove node
    for (NodePtr removeNode : inAnchorRelatedInfo.removeNode) {
      if (removeNode->GetType() == kOpTypeData) {
        OP_LOGI(removeNode->GetType().c_str(), "Data node[%s] cannot be removed.", removeNode->GetName().c_str());
        continue;
      }
      FUSION_PASS_CHECK(
          ge::GRAPH_SUCCESS != graph.RemoveNode(removeNode),
          VECTOR_FUSION_INNER_ERR_REPORT("HandleUnlinkedInAnchor", "Fail to remove Node[%s] while handling unlinked inDataAnchor.",
                  removeNode->GetName().c_str()),
          return FAILED);
    }
  }
  return SUCCESS;
}

bool RemoveNodeFusionPass::VerifyRemoveInAnchor(InDataAnchorPtr inDataAnchorPtr,
                                                InAnchorRelatedInfo& relatedInfo) {
  if (inDataAnchorPtr == nullptr || inDataAnchorPtr->GetPeerOutAnchor() == nullptr) {
    return true;
  }

  OutDataAnchorPtr outDataAnchorPtr = inDataAnchorPtr->GetPeerOutAnchor();
  relatedInfo.removeEdgeVec.push_back({inDataAnchorPtr, outDataAnchorPtr});

  // if inDataAnchorPtr has more than one out data anchor, just remove the edge.
  if (outDataAnchorPtr->GetPeerInDataAnchors().size() > PASS_UINT_ONE) {
    return true;
  }

  // if the node of other end has other out data anchor,
  // then this node cannot be removed
  NodePtr nodeOfOtherEnd = outDataAnchorPtr->GetOwnerNode();
  uint32_t outAnchorLinkedCount = 0;
  for (OutDataAnchorPtr outAnchorPtr : nodeOfOtherEnd->GetAllOutDataAnchors()) {
    if (!outAnchorPtr->GetPeerInDataAnchors().empty()) {
      outAnchorLinkedCount++;
    }
  }
  FUSION_PASS_CHECK(outAnchorLinkedCount > PASS_UINT_ONE,
                    OP_LOGD(nodeOfOtherEnd->GetType().c_str(),
                            "Node[%s] cannot be removed "
                            "for it has more than one out data anchor",
                            nodeOfOtherEnd->GetName().c_str()),
                    return false);

  relatedInfo.removeNode.push_back(nodeOfOtherEnd);
  // if node has only one out data anchor,
  // then check whether the node has in data anchor
  vector<InDataAnchorPtr> linkedInAnchorVec;
  for (InDataAnchorPtr inAnchorPtr : nodeOfOtherEnd->GetAllInDataAnchors()) {
    if (inAnchorPtr->GetPeerOutAnchor() != nullptr) {
      linkedInAnchorVec.push_back(inAnchorPtr);
    }
  }
  if (linkedInAnchorVec.empty()) {
    OP_LOGD(nodeOfOtherEnd->GetType().c_str(), "The node[%s] does not have any linked in data anchor.",
            nodeOfOtherEnd->GetName().c_str());
    return true;
  }

  // if this node have some linked in data anchor
  // all the inDataAnchors should also be verified
  for (InDataAnchorPtr inAnchorPtr : linkedInAnchorVec) {
    if (!VerifyRemoveInAnchor(inAnchorPtr, relatedInfo)) {
      return false;
    }
  }
  OP_LOGD(nodeOfOtherEnd->GetType().c_str(),
          "The node[%s] can be removed "
          "for all in data anchors of this node can be removed.",
          nodeOfOtherEnd->GetName().c_str());
  return true;
}

Status RemoveNodeFusionPass::ReLinkControlAnchor(NodePtr node) const {
  // in control anchor link to all output nodes
  InControlAnchorPtr inControlAnchorPtr = node->GetInControlAnchor();
  if (inControlAnchorPtr != nullptr) {
    if (!inControlAnchorPtr->GetPeerOutControlAnchors().empty() && !node->GetOutDataNodes().empty()) {
      for (OutControlAnchorPtr outControlAnchorPtr : inControlAnchorPtr->GetPeerOutControlAnchors()) {
        for (NodePtr outNode : node->GetOutDataNodes()) {
          if (outNode->GetInControlAnchor() == nullptr) {
            continue;
          }
          FUSION_PASS_CHECK(AddCtrlEdge(outControlAnchorPtr, outNode->GetInControlAnchor()) != SUCCESS,
                            VECTOR_FUSION_INNER_ERR_REPORT(outNode->GetType().c_str(), "Fail to re-link input control edge."), return FAILED);
        }
      }
    }
    inControlAnchorPtr->UnlinkAll();
  }

  // out control anchor link to all input nodes
  OutControlAnchorPtr outControlAnchorPtr = node->GetOutControlAnchor();
  if (outControlAnchorPtr != nullptr) {
    if (!outControlAnchorPtr->GetPeerInControlAnchors().empty() && !node->GetInDataNodes().empty()) {
      for (InControlAnchorPtr inControlAnchorPtr : outControlAnchorPtr->GetPeerInControlAnchors()) {
        for (NodePtr inNode : node->GetInDataNodes()) {
          if (inNode->GetOutControlAnchor() == nullptr) {
            continue;
          }

          FUSION_PASS_CHECK(AddCtrlEdge(inNode->GetOutControlAnchor(), inControlAnchorPtr) != SUCCESS,
                            VECTOR_FUSION_INNER_ERR_REPORT(inNode->GetType().c_str(), "Fail to re-link output control edge."), return FAILED);
        }
      }
    }
    outControlAnchorPtr->UnlinkAll();
  }

  return SUCCESS;
}

Status RemoveNodeFusionPass::AddCtrlEdge(OutControlAnchorPtr outCtrlAnchorPtr,
                                         InControlAnchorPtr inCtrlAnchorPtr) const {
  // check whether this control anchor has been linked.
  NodePtr otherEndNode = inCtrlAnchorPtr->GetOwnerNode();
  bool isLinked = false;
  for (InControlAnchorPtr inCtrlAnchorPtr : outCtrlAnchorPtr->GetPeerInControlAnchors()) {
    if (otherEndNode == inCtrlAnchorPtr->GetOwnerNode()) {
      isLinked = true;
      break;
    }
  }
  if (isLinked) {
    return SUCCESS;
  }
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outCtrlAnchorPtr, inCtrlAnchorPtr) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT("AddCtrlEdge", "Fail to add ctrl edge."), return FAILED);
  return SUCCESS;
}

Status RemoveNodeFusionPass::ReLinkDataAnchor(vector<LinkIndexPair>& linkIndexPairVec,
                                              map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                                              map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
                                              map<uint32_t, OutDataAnchorPtr>& inputOutDataAnchorMap,
                                              map<uint32_t, vector<InDataAnchorPtr>>& outputInDataAnchorMap) {
  // unlink all edge of node
  map<uint32_t, InDataAnchorPtr>::iterator inIter;
  map<uint32_t, OutDataAnchorPtr>::iterator outIter;
  for (inIter = inDataAnchorMap.begin(); inIter != inDataAnchorMap.end(); inIter++) {
    inIter->second->UnlinkAll();
  }
  for (outIter = outDataAnchorMap.begin(); outIter != outDataAnchorMap.end(); outIter++) {
    outIter->second->UnlinkAll();
  }

  // link the edge
  map<uint32_t, OutDataAnchorPtr>::iterator inputIter;
  map<uint32_t, vector<InDataAnchorPtr>>::iterator outputIter;
  for (LinkIndexPair linkPair : linkIndexPairVec) {
    OP_LOGD("ReLinkDataAnchor", "Re-link data anchor [%u, %u].", linkPair.inAnchorIndex, linkPair.outAnchorIndex);
    inputIter = inputOutDataAnchorMap.find(linkPair.inAnchorIndex);
    outputIter = outputInDataAnchorMap.find(linkPair.outAnchorIndex);
    if (inputIter == inputOutDataAnchorMap.end() || outputIter == outputInDataAnchorMap.end()) {
      continue;
    }

    OutDataAnchorPtr inputOutDataAnchor = inputIter->second;
    vector<InDataAnchorPtr> outInputDataAnchorVec = outputIter->second;
    if (inputOutDataAnchor == nullptr || outInputDataAnchorVec.empty()) {
      continue;
    }

    for (InDataAnchorPtr inAnchor : outInputDataAnchorVec) {
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(inputOutDataAnchor, inAnchor) != ge::GRAPH_SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT("ReLinkDataAnchor", "Fail to link data edge[%u, %u] of the node.",
                                linkPair.inAnchorIndex, linkPair.outAnchorIndex),
                        return FAILED);
    }
  }
  return SUCCESS;
}

Status GatherNdPreCheck(NodePtr node) {
  string curOpType = node->GetType();
  OP_LOGI(curOpType.c_str(), "Current Node name is :%s", node->GetName().c_str());
  size_t gather_nd_input_size = node->GetOpDesc()->GetInputsSize();
  if (gather_nd_input_size != 2) {
    OP_LOGW(curOpType.c_str(), "gather_nd_input_size is %u which must be 2, pre check failed", gather_nd_input_size);
    return NOT_CHANGED;
  }
  ge::GeTensorDesc gather_nd_indices = node->GetOpDesc()->GetInputDesc(1);
  ge::GeShape indice_shape = gather_nd_indices.GetShape();
  vector<int64_t> dim_info = indice_shape.GetDims();
  if (dim_info.size() == 0) {
    // case []
    OP_LOGW(curOpType.c_str(), "The shape_indice did not support []");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(dim_info[dim_info.size() - 1]) ||
      dim_info[dim_info.size() - 1] != 0) {
    // case [3,2]; [0,0,3,2]; ...
    OP_LOGI(curOpType.c_str(), "Indice of GatherNd gets a normal tensor or "
                               "ARemoveNodeFusionPass cannot be applied for unknown shape");
    return NOT_CHANGED;
  }
  for (int i = (int)dim_info.size() - 2; i >= 0; i--) {
    // case [2,0], [1,1,0,0]
    if (PatternFusionUtil::IsUnknownShape(dim_info[i]) || dim_info[i] != 1) {
      OP_LOGI(curOpType.c_str(), "Indice of GatherNd gets a normal tensor or "
                                 "ARemoveNodeFusionPass cannot be applied for unknown shape");
      return NOT_CHANGED;
    }
  }
  // case [1,1,0]
  OP_LOGI(curOpType.c_str(), "Indice of GatherNd gets a zero-tensor");
  return SUCCESS;
}

REGISTER_REMOVENODE("GatherNd").SetPreCheckFunc(GatherNdPreCheck).AddLinkPair(0, 0);
}  // namespace fe
