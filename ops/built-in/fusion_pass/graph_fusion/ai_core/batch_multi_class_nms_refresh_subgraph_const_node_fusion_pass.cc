/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file batch_multi_class_nms_refresh_output_dtype_fusion_pass.cpp
 * \brief batch_multi_class_nms fusion pass
 */
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "op_log.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "batch_multi_class_nms_refresh_subgraph_const_node_fusion_pass.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppressionRefreshSubgraphConstNode";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";

vector<FusionPattern*> BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::ReplaceNodeWithNewNode(
    ge::ComputeGraphPtr& subgraph, ge::NodePtr& oldNode, ge::NodePtr& newNode) {
  for (auto& outputAnchor : oldNode->GetAllOutDataAnchors()) {
    for (auto& peerInDataAnchor : outputAnchor->GetPeerInDataAnchors()) {
      if (peerInDataAnchor == nullptr) {
        continue;
      }
      FUSION_PASS_CHECK(peerInDataAnchor->ReplacePeer(outputAnchor, newNode->GetOutDataAnchor(0)) != ge::GRAPH_SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "ReplaceNodeWithNewNode failed."), return FAILED);
    }
  }
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveNodeWithoutRelink(subgraph, oldNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNodeWithoutRelink failed."), return FAILED);
  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::UpdateConstNode(
    ge::ComputeGraphPtr& subgraph, map<int64_t, ge::NodePtr>& constNodeMap, int64_t maxTotalSize) {
  // Get ATTR from graph which named "_subgraph_real_dims"
  vector<int64_t> realDims;
  ge::AttrUtils::GetListInt(subgraph, "_subgraph_real_dims", realDims);

  vector<NodePtr> dataNodes;
  for (auto& node : subgraph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      dataNodes.emplace_back(node);
    }
  }
  for (auto& node : dataNodes) {
    int32_t parentNodeIndex = 0;
    ge::AttrUtils::GetInt(node->GetOpDesc(), "_parent_node_index", parentNodeIndex);
    if (constNodeMap.find(parentNodeIndex) != constNodeMap.end()) {
      vector<GeTensorPtr> weightsVector = ge::OpDescUtils::MutableWeights(constNodeMap.find(parentNodeIndex)->second);
      GeTensorPtr weights = weightsVector[0];
      auto newWeight =
          make_shared<GeTensor>(weights->GetTensorDesc(), weights->GetData().GetData(), weights->GetData().size());
      int32_t* shapeData = reinterpret_cast<int32_t*>(newWeight->MutableData().GetData());
      size_t shapeDataSize = newWeight->GetData().GetSize() / sizeof(int32_t);
      bool needChange = false;
      for (size_t shapeIndex = 0; shapeIndex < shapeDataSize; ++shapeIndex) {
        if (shapeData[shapeIndex] == maxTotalSize) {
          shapeData[shapeIndex] = realDims[0];
          needChange = true;
          break;
        }
      }
      if (needChange) {
        // New a const node and set as weights
        OpDescPtr newConstOpDesc = ge::AttrUtils::CopyOpDesc(constNodeMap.find(parentNodeIndex)->second->GetOpDesc());
        FUSION_PASS_CHECK(newConstOpDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The newConstOpDesc is null."),
                          return FAILED);
        newConstOpDesc->SetName(newConstOpDesc->GetName() + subgraph->GetName());
        NodePtr newConstNode = subgraph->AddNode(newConstOpDesc);
        FUSION_PASS_CHECK(newConstNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The newConstNode is null."),
                          return FAILED);
        ge::OpDescUtils::SetWeights(newConstNode, {newWeight});
        FUSION_PASS_CHECK(ReplaceNodeWithNewNode(subgraph, node, newConstNode) == FAILED,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "ReplaceNodeWithNewNode failed."), return FAILED);
      }
    }
  }
  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::AddPadDNode(
    ge::ComputeGraphPtr& subgraph, int32_t index, vector<vector<int64_t>> paddingVector) {
  auto netOutput = subgraph->FindFirstNodeMatchType("NetOutput");
  auto node = netOutput->GetInDataNodes().at(index);

  // Add Pad_D
  OpDescPtr padDescPtr = ge::AttrUtils::CloneOpDesc(node->GetOpDesc());
  FUSION_PASS_CHECK(padDescPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The padDescPtr is null."), return FAILED);
  padDescPtr->SetType("PadD");
  padDescPtr->SetName(padDescPtr->GetName() + "_PadD");

  map<string, uint32_t> outputNameIdx;
  outputNameIdx["y"] = 0;
  padDescPtr->UpdateOutputName(outputNameIdx);

  int tmpInputSize = padDescPtr->GetInputsSize();
  while (tmpInputSize > 0) {
    tmpInputSize--;
    ge::OpDescUtils::ClearInputDesc(padDescPtr, tmpInputSize);
  }
  GeTensorDesc padInputTensorDesc = node->GetOpDesc()->GetOutputDesc(0);
  padDescPtr->AddInputDesc(0, padInputTensorDesc);
  ge::AttrUtils::SetListListInt(padDescPtr, "paddings", paddingVector);

  NodePtr padNodePtr = subgraph->AddNode(padDescPtr);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node->GetOutDataAnchor(0), netOutput->GetInDataAnchor(index)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveEdge failed."), return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(padNodePtr->GetOutDataAnchor(0), netOutput->GetInDataAnchor(index)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), padNodePtr->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed."), return FAILED);
  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::UpdateOutput(ge::ComputeGraphPtr& subgraph,
                                                                                        int64_t maxTotalSize) {
  vector<int64_t> realDims;
  ge::AttrUtils::GetListInt(subgraph, "_subgraph_real_dims", realDims);

  auto netOutput = subgraph->FindFirstNodeMatchType("NetOutput");

  vector<vector<vector<int64_t>>> paddingList = {{{0, 0}, {0, maxTotalSize - realDims[0]}},
                                                 {{0, 0}, {0, maxTotalSize - realDims[0]}, {0, 0}},
                                                 {{0, 0}, {0, maxTotalSize - realDims[0]}, {0, 0}, {0, 0}}};
  for (size_t outPutIndex = 0; outPutIndex < netOutput->GetInDataNodes().size(); ++outPutIndex) {
    if (netOutput->GetInDataNodes().at(outPutIndex)->GetName().find("scores") != string::npos ||
        netOutput->GetInDataNodes().at(outPutIndex)->GetName().find("classes") != string::npos) {
      FUSION_PASS_CHECK(AddPadDNode(subgraph, outPutIndex, paddingList[0]) == FAILED,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddPadDNode failed."), return FAILED);
    } else if (netOutput->GetInDataNodes().at(outPutIndex)->GetName().find("boxes") != string::npos) {
      FUSION_PASS_CHECK(AddPadDNode(subgraph, outPutIndex, paddingList[1]) == FAILED,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddPadDNode failed."), return FAILED);
    } else if (netOutput->GetInDataNodes().at(outPutIndex)->GetName().find("masks") != string::npos) {
      FUSION_PASS_CHECK(AddPadDNode(subgraph, outPutIndex, paddingList[2]) == FAILED,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddPadDNode failed."), return FAILED);
    } else {
      continue;
    }
  }
  return SUCCESS;
}

void BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::GetNMSNodeIndex(ge::ComputeGraph& graph,
                                                                                         const ge::NodePtr& checkNode,
                                                                                         int64_t& nmsNodeIndex) {
  for (const auto& node : graph.GetDirectNode()) {
    if (node->GetType() == "BatchMultiClassNonMaxSuppression") {
      nmsNodeIndex++;
      if (node->GetOpDesc()->GetName() == checkNode->GetOpDesc()->GetName()) {
        break;
      }
    }
  }
}

Status BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass::Fusion(ge::ComputeGraph& graph,
                                                                                  Mapping& mapping,
                                                                                  vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "Define BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  int64_t nmsNodeIndex = 0;
  GetNMSNodeIndex(graph, fusedNode, nmsNodeIndex);
  if (nmsNodeIndex == 2U) {
    int64_t maxTotalSize;
    ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "max_total_size", maxTotalSize);
    ge::NodePtr partitionalCall = nullptr;
    for (auto& node : fusedNode->GetOutDataNodes()) {
      if (node->GetType() == "PartitionedCall") {
        partitionalCall = node;
      }
    }
    FUSION_PASS_CHECK(partitionalCall == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The partitionalCall is null, fusion failed."),
                      return PARAM_INVALID);
    // Find case subgraph
    auto caseSubgraph = ge::NodeUtils::GetSubgraph(*partitionalCall, 0);
    FUSION_PASS_CHECK(caseSubgraph == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The caseSubgraph is null, fusion failed."), return PARAM_INVALID);
    ge::NodePtr caseNode = caseSubgraph->FindFirstNodeMatchType("Case");
    FUSION_PASS_CHECK(caseNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The caseNode is null, fusion failed."),
                      return PARAM_INVALID);
    // Get const nodeMapList
    map<int64_t, ge::NodePtr> constNodeMap;
    for (size_t inNodeIndex = 0; inNodeIndex < caseNode->GetInDataNodes().size(); ++inNodeIndex) {
      auto inNode = caseNode->GetInDataNodes().at(inNodeIndex);
      auto realInNode = ge::NodeUtils::GetInNodeCrossSubgraph(inNode);
      if (realInNode->GetType() == "Const") {
        constNodeMap[inNodeIndex] = realInNode;
      }
    }
    // From case get real subgraph, then traverse each subgraph and update const value
    OpDescPtr caseDesc = caseNode->GetOpDesc();
    for (auto& nameIter : caseDesc->GetSubgraphInstanceNames()) {
      auto subgraph = graph.GetSubgraph(nameIter);
      FUSION_PASS_CHECK(subgraph == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The subgraph is null, fusion failed."),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(UpdateConstNode(subgraph, constNodeMap, maxTotalSize) == FAILED,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateConstNode failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(UpdateOutput(subgraph, maxTotalSize) == FAILED,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutput failed, fusion failed."), return FAILED);
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "Define BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass", BUILT_IN_AFTER_MULTI_DIMS_PASS,
              BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass);
}  // namespace fe
