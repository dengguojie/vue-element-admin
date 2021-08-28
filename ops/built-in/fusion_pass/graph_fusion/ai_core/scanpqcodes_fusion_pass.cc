/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file scanpqcodes_fusion_pass.cc
 * \brief ScanPQCodes fusion pass(Parallel ScanPQCodes)
 */
#include <stdint.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "scanpqcodes_fusion_pass.h"

namespace fe {

static const std::string PATTERN_SCAN_PQ_CODES = "ScanPQCodes";
static const std::string PATTERN_TOPK_PQ_DISTANCE = "TopKPQDistance";
static const std::string OP_TYPE_SCAN_PQ_CODES = "ScanPQCodes";
static const std::string OP_TYPE_TOPK_PQ_DISTANCE = "TopKPQDistance";
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_SPLIT_COUNT = "split_count";
static const std::string ATTR_SPLIT_INDEX = "split_index";
static const uint32_t SCANPQ_INPUT_SIZE = 5;

/*!
 * @brief Define ScanPQCodes -> TopKPQDistance pattern.
 * The graph struct need to adapt is shown as follows:
 *
 *                bucket_base_distance
 *                          |
 *           bucket_list    |  bucket_limits
 *                    \     |     /
 *                ivf  \    |    / bucket_offsets
 *                   \  \   |   /  /
 *                    \  \  |  /  / adc_tables
 *                     \  \ | /  /  /
 *                      ScanPQCodes
 *                          |
 *                    TopKPQDistance
 *                          |
 *                       output
 *
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ScanPQCodesFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string pass_name = "ScanPQCodesFusionPass";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_SCAN_PQ_CODES, {OP_TYPE_SCAN_PQ_CODES})
      .AddOpDesc(PATTERN_TOPK_PQ_DISTANCE, {OP_TYPE_TOPK_PQ_DISTANCE})
      .SetInputs(PATTERN_TOPK_PQ_DISTANCE, {PATTERN_SCAN_PQ_CODES})
      .SetOutput(PATTERN_TOPK_PQ_DISTANCE);
  patterns.push_back(pattern);
  return patterns;
}

Status ScanPQCodesFusionPass::SplitScanPQCodesNode(ge::ComputeGraph& graph, ge::NodePtr& scanPQCodesNode,
                                                   ge::NodePtr& scanPQCodesNodeAicore,
                                                   ge::NodePtr& scanPQCodesNodeVectorCore) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into split ScanPQCodes node.");
  ge::OpDescPtr scanPQCodesDesc = scanPQCodesNode->GetOpDesc();
  FUSION_PASS_CHECK(scanPQCodesDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get OpDesc of ScanPQCodes."),
                    return PARAM_INVALID);

  ge::OpDescPtr scanPQCodesDescAic = AttrUtils::CloneOpDesc(scanPQCodesDesc);
  FUSION_PASS_CHECK(scanPQCodesDescAic == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Faile to create ScanPQCodes(AI Core) op desc."), return FAILED);
  scanPQCodesDescAic->SetName(scanPQCodesDesc->GetName() + "_aicore");

  // Set attribute of AICore flag.
  ge::AttrUtils::SetStr(scanPQCodesDescAic, ATTR_OP_SPECIFIED_ENGINE_NAME, "AIcoreEngine");
  ge::AttrUtils::SetStr(scanPQCodesDescAic, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "AIcoreEngine");
  ge::AttrUtils::SetInt(scanPQCodesDescAic, ATTR_SPLIT_COUNT, (int64_t)2);
  ge::AttrUtils::SetInt(scanPQCodesDescAic, ATTR_SPLIT_INDEX, 0);

  scanPQCodesNodeAicore = graph.AddNode(scanPQCodesDescAic);
  FUSION_PASS_CHECK(scanPQCodesNodeAicore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ScanPQCodes(AI Core) to graph"),
                    return FAILED);

  ge::OpDescPtr scanPQCodesDescVec = AttrUtils::CloneOpDesc(scanPQCodesDesc);
  FUSION_PASS_CHECK(scanPQCodesDescVec == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Faile to create ScanPQCodes(Vector Core) op desc."), return FAILED);
  scanPQCodesDescVec->SetName(scanPQCodesDesc->GetName() + "_vectorcore");

  // Set attribute of VectorCore flag.
  ge::AttrUtils::SetStr(scanPQCodesDescVec, ATTR_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  ge::AttrUtils::SetStr(scanPQCodesDescVec, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");
  ge::AttrUtils::SetInt(scanPQCodesDescVec, ATTR_SPLIT_COUNT, (int64_t)2);
  ge::AttrUtils::SetInt(scanPQCodesDescVec, ATTR_SPLIT_INDEX, (int64_t)1);

  scanPQCodesNodeVectorCore = graph.AddNode(scanPQCodesDescVec);
  FUSION_PASS_CHECK(scanPQCodesNodeVectorCore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ScanPQCodes(Vector Core) to graph"),
                    return FAILED);
  uint32_t index = 0;
  for (auto inputDesc : scanPQCodesDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(scanPQCodesNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                                         scanPQCodesNodeAicore->GetInDataAnchor(index)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Failed to add edge from ScanPQCode node to ScanPQCodes(AI "
                              "Core) node."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(scanPQCodesNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                                         scanPQCodesNodeVectorCore->GetInDataAnchor(index)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Failed to add edge from ScanPQCode node to ScanPQCodes(Vector "
                              "Core) node."),
                      return FAILED);
    index++;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to split ScanPQCodes node.");
  return SUCCESS;
}

Status ScanPQCodesFusionPass::CreateNewTopKPQDistanceNode(ge::ComputeGraph& graph, ge::NodePtr& scanPQCodesNode,
                                                          vector<ge::NodePtr>& newScanPQCodesNodes,
                                                          ge::NodePtr& topKPQDistanceNode,
                                                          ge::NodePtr& fusedTopKPQDistanceNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into CreateNewTopKPQDistanceNode function.");
  ge::OpDescPtr topKPQDistanceNodeDesc = topKPQDistanceNode->GetOpDesc();
  FUSION_PASS_CHECK(topKPQDistanceNodeDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get OpDesc of topKPQDistanceNode."),
                    return PARAM_INVALID);

  ge::OpDescPtr fusedTopKPQDistanceNodeDesc = AttrUtils::CloneOpDesc(topKPQDistanceNodeDesc);
  FUSION_PASS_CHECK(fusedTopKPQDistanceNodeDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get OpDesc of FusedTopKPQDistanceNodeDesc."),
                    return PARAM_INVALID);
  fusedTopKPQDistanceNodeDesc->SetName(topKPQDistanceNodeDesc->GetName() + "/fused");
  for (int32_t index = fusedTopKPQDistanceNodeDesc->GetAllInputsDesc().size() - 1; index >= 0; index--) {
    FUSION_PASS_CHECK(!ge::OpDescUtils::ClearInputDesc(fusedTopKPQDistanceNodeDesc, index),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's clear %d th input failed.",
                              fusedTopKPQDistanceNodeDesc->GetName().c_str(), index),
                      return PARAM_INVALID);
  }
  vector<string> inputsName = {"actual_count", "pq_distance", "grouped_extreme_distance", "pq_ivf", "pq_index"};
  uint32_t inputDescSize = topKPQDistanceNodeDesc->GetAllInputsDesc().size();
  FUSION_PASS_CHECK(inputDescSize != SCANPQ_INPUT_SIZE,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get AllInputsDesc of FusedTopKPQDistanceNodeDesc."),
                    return PARAM_INVALID);

  uint32_t index = 0;
  for (auto inputDesc : topKPQDistanceNodeDesc->GetAllInputsDesc()) {
    ge::GeTensorDesc inputTensorDesc = topKPQDistanceNodeDesc->GetInputDesc(index).Clone();
    fusedTopKPQDistanceNodeDesc->AddInputDesc(inputsName[index] + std::to_string(0), inputTensorDesc);
    fusedTopKPQDistanceNodeDesc->AddInputDesc(inputsName[index] + std::to_string(1), inputTensorDesc);
    index++;
  }
  fusedTopKPQDistanceNode = graph.AddNode(fusedTopKPQDistanceNodeDesc);

  index = 0;
  for (auto inputDesc : topKPQDistanceNodeDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(newScanPQCodesNodes[0]->GetOutDataAnchor(index),
                                fusedTopKPQDistanceNode->GetInDataAnchor(index * 2)) != ge::GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add input edge between %s and %s %d th failed.",
                newScanPQCodesNodes[0]->GetName().c_str(), fusedTopKPQDistanceNode->GetName().c_str(), index),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(newScanPQCodesNodes[1]->GetOutDataAnchor(index),
                                fusedTopKPQDistanceNode->GetInDataAnchor(index * 2 + 1)) != ge::GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add input edge between %s and %s %d th failed.",
                newScanPQCodesNodes[1]->GetName().c_str(), fusedTopKPQDistanceNode->GetName().c_str(), index),
        return NOT_CHANGED);
    index++;
  }

  index = 0;
  for (auto outAnchor : topKPQDistanceNode->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr inAnchorPtr : outAnchor->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedTopKPQDistanceNode->GetOutDataAnchor(index), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from %s to fusion node %s's %d th failed.",
                  topKPQDistanceNode->GetName().c_str(), fusedTopKPQDistanceNode->GetName().c_str(), index),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from %s to fusion node %s's %d index success.",
              topKPQDistanceNode->GetName().c_str(), fusedTopKPQDistanceNode->GetName().c_str(), index);
    }
    index++;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End of CreateNewTopKPQDistanceNode.");
  return SUCCESS;
}

Status ScanPQCodesFusionPass::ClearFusedNode(ge::ComputeGraph& graph, ge::NodePtr& node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into clear fused node.");

  std::string nodeName = node->GetName();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Node name: %s", nodeName.c_str());
  for (auto inAnchor : node->GetAllInDataAnchors()) {
    if (inAnchor == nullptr) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to UnlinkAll of %s for inAnchor is nullptr.", nodeName.c_str());
    } else {
      inAnchor->UnlinkAll();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Finished to UnlinkAll inDataAnchor of node[%s].", nodeName.c_str());
    }
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to remove node[%s].", nodeName.c_str()), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to clear fused node.");
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] new_nodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ScanPQCodesFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ScanPQCodes fusion pass.");

  ge::NodePtr scanPQCodesNode = GetNodeFromMapping(PATTERN_SCAN_PQ_CODES, mapping);
  ge::NodePtr topKPQDistanceNode = GetNodeFromMapping(PATTERN_TOPK_PQ_DISTANCE, mapping);
  FUSION_PASS_CHECK(scanPQCodesNode == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to get ScanPQCodes Node."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(topKPQDistanceNode == nullptr,
                    CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to get TopKPQDistance Node."), return PARAM_INVALID);
  ge::NodePtr scanPQCodesNodeAicore = nullptr;
  ge::NodePtr scanPQCodesNodeVectorCore = nullptr;
  ge::NodePtr fusedTopKPQDistanceNode = nullptr;
  Status scanPQCodesRet =
      SplitScanPQCodesNode(graph, scanPQCodesNode, scanPQCodesNodeAicore, scanPQCodesNodeVectorCore);
  FUSION_PASS_CHECK(SUCCESS != scanPQCodesRet, OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to split scan_pq_codes node."),
                    return scanPQCodesRet);
  vector<ge::NodePtr> newScanPQCodesNodes = {scanPQCodesNodeAicore, scanPQCodesNodeVectorCore};

  Status topKDistanceRet = CreateNewTopKPQDistanceNode(graph, scanPQCodesNode, newScanPQCodesNodes, topKPQDistanceNode,
                                                       fusedTopKPQDistanceNode);
  FUSION_PASS_CHECK(SUCCESS != topKDistanceRet,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add new TopKPQDistanceNode node."),
                    return topKDistanceRet);

  Status clearFusedNodeRet = ClearFusedNode(graph, scanPQCodesNode);
  FUSION_PASS_CHECK(SUCCESS != clearFusedNodeRet,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to clear ScanPQCodesNode node."), return clearFusedNodeRet);
  clearFusedNodeRet = ClearFusedNode(graph, topKPQDistanceNode);
  FUSION_PASS_CHECK(SUCCESS != clearFusedNodeRet,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to clear TopKPQDistanceNode nodes."),
                    return clearFusedNodeRet);
  new_nodes.push_back(scanPQCodesNodeAicore);
  new_nodes.push_back(scanPQCodesNodeVectorCore);
  new_nodes.push_back(fusedTopKPQDistanceNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ScanPQCodes fusion pass.");
  return SUCCESS;
}
REGISTER_PASS("ScanPQCodesFusionPass", BUILT_IN_GRAPH_PASS, ScanPQCodesFusionPass);
}  // namespace fe
