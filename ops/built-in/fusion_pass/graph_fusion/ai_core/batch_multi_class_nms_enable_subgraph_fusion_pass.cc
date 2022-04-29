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
 * \file batch_multi_class_nms_enable_subgraph_fusion_pass.cpp
 * \brief batch_multi_class_nms_enable_subgraph fusion pass
 */
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "batch_multi_class_nms_enable_subgraph_fusion_pass.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppressionEnableSubgraph";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";
static const string ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_SHAPE = "_subgraph_multi_dims_input_shape";
static const string ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_DYN_DIMS = "_subgraph_multi_dims_input_dims";
static const string ATTR_SUBGRAPH_DYNAMIC_DIMS_INDEX = "_subgraph_multi_dims_index";
static const int64_t SPLIT_LEVEL = 8;

vector<FusionPattern*> BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass::GetNMSNodeIndex(ge::ComputeGraph& graph,
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

string SubgraphDimsRepeat(const string& target, unsigned repeatTimes) {
  string ret;
  ret.reserve(target.size() * repeatTimes);
  while (repeatTimes--) {
    ret += target;
  }
  return ret;
}

Status BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass::AddNMSSubgraphAttr(ge::NodePtr& node,
                                                                                    int64_t batchsize,
                                                                                    int64_t maxTotalSize) {
  // Add ATTR to the first following nodes in subgraph
  string subgraphShape = "";
  string subgraphDims = "";
  int64_t inputCount = 0;
  uint32_t levelNum = maxTotalSize / SPLIT_LEVEL + 1;
  // Update subgraph shape
  for (auto inAnchor : node->GetAllInDataAnchors()) {
    auto outAnchor = inAnchor->GetPeerOutAnchor();
    if (outAnchor == nullptr) {
      continue;
    }
    if (outAnchor->GetOwnerNode()->GetType() == "BatchMultiClassNonMaxSuppression") {
      if (outAnchor->GetIdx() == 0) {
        subgraphShape += to_string(inAnchor->GetIdx()) + ":" + to_string(batchsize) + ",-1,4;";
        inputCount++;
      } else {
        subgraphShape += to_string(inAnchor->GetIdx()) + ":" + to_string(batchsize) + ",-1;";
        inputCount++;
      }
    }
  }
  // Update subgraphDims
  for (uint32_t dimIndex = 1; dimIndex <= levelNum; ++dimIndex) {
    if (dimIndex * SPLIT_LEVEL < maxTotalSize) {
      subgraphDims +=
          SubgraphDimsRepeat(to_string(dimIndex * SPLIT_LEVEL) + ",", inputCount - 1) + to_string(dimIndex * SPLIT_LEVEL) + ";";
    } else {
      subgraphDims += SubgraphDimsRepeat(to_string(maxTotalSize) + ",", inputCount - 1) + to_string(maxTotalSize) + ";";
    }
  }
  subgraphShape.pop_back();
  subgraphDims.pop_back();

  FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(node->GetOpDesc(), ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_SHAPE, subgraphShape),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_SHAPE failed, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetStr(node->GetOpDesc(), ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_DYN_DIMS, subgraphDims),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set ATTR_SUBGRAPH_DYNAMIC_DIMS_INPUT_DYN_DIMS failed, fusion failed."),
      return FAILED);

  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass::UpdateSubgraphAttribute(ge::NodePtr& startNode,
                                                                                         int64_t nmsNodeIndex) {
  queue<ge::NodePtr> nodeQueue;
  nodeQueue.push(startNode);
  while (!nodeQueue.empty()) {
    auto subgraphNode = nodeQueue.front();
    if (subgraphNode->GetType() == "NetOutput" || subgraphNode->GetType() == "BatchMultiClassNonMaxSuppression") {
      nodeQueue.pop();
      continue;
    }
    // Add child node to nodeQueue
    for (unsigned int outAnchorIndex = 0; outAnchorIndex < subgraphNode->GetAllOutDataAnchors().size();
         ++outAnchorIndex) {
      OutDataAnchorPtr subgraphOutAnchor = subgraphNode->GetOutDataAnchor(outAnchorIndex);
      for (auto subgraphInAnchor : subgraphOutAnchor->GetPeerInDataAnchors()) {
        nodeQueue.push(subgraphInAnchor->GetOwnerNode());
      }
    }
    // Add
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(subgraphNode->GetOpDesc(), ATTR_SUBGRAPH_DYNAMIC_DIMS_INDEX, nmsNodeIndex),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set ATTR_SUBGRAPH_DYNAMIC_DIMS_INDEX failed, fusion failed."),
                      return FAILED);
    nodeQueue.pop();
  }
  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                                        vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The fusedNode is null, fusion failed."),
                    return FAILED);

  int64_t nmsNodeIndex = 0;
  GetNMSNodeIndex(graph, fusedNode, nmsNodeIndex);
  // Only handle the second nmsNode
  if (nmsNodeIndex == 2U) {
    // from DataNode get batchsize
    ge::NodePtr dataNode = graph.FindFirstNodeMatchType("Data");
    FUSION_PASS_CHECK(dataNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Get node Data failed, fusion failed."),
                      return FAILED);
    int64_t batchsize = dataNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDims()[0];
    int64_t maxTotalSize;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "max_total_size", maxTotalSize),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Get max_total_size failed, fusion failed."), return FAILED);
    // Get the first direct output node of nms
    for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
      if (outAnchor->GetIdx() == 3U) {
        continue;
      }
      for (auto inAnchor : outAnchor->GetPeerInDataAnchors()) {
        auto node = inAnchor->GetOwnerNode();
        FUSION_PASS_CHECK(AddNMSSubgraphAttr(node, batchsize, maxTotalSize) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "AddNMSSubgraphAttr failed, fusion failed."), return FAILED);
        FUSION_PASS_CHECK(UpdateSubgraphAttribute(node, nmsNodeIndex) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateSubgraphAttribute failed, fusion failed."),
                          return FAILED);
      }
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass", BUILT_IN_PREPARE_GRAPH_PASS,
              BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass);
}  // namespace fe