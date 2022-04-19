/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file relu_cast_fusion_pass.cc
 * \brief relu cast fusion pass
 */
#include "relu_cast_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_CONCATV2D = "ConcatV2D";
static const std::string PATTERN_TRANSDATA = "TransData";
static const std::string PATTERN_RELU = "Relu";
static const std::string PATTERN_CAST = "Cast";
static const std::string OP_TYPE_CONCATV2D = "ConcatV2D";
static const std::string OP_TYPE_TRANSDATA = "TransData";
static const std::string OP_TYPE_CAST = "Cast";
static const std::string OP_TYPE_RELU = "Relu";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        ConcatV2D                ConcatV2D
 *            |                        |
 *        TransData                TransData
 *            |           ==>          |
 *           Relu                     Cast
 *            |                        |
 *           Cast                     Relu
 *
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ReluCastFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into define ReluCastFusionPass pattern");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReluCastFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_CONCATV2D, {OP_TYPE_CONCATV2D})
      .AddOpDesc(PATTERN_TRANSDATA, {OP_TYPE_TRANSDATA})
      .AddOpDesc(PATTERN_RELU, {OP_TYPE_RELU})
      .AddOpDesc(PATTERN_CAST, {OP_TYPE_CAST})
      .SetInputs(PATTERN_TRANSDATA, {PATTERN_CONCATV2D})
      .SetInputs(PATTERN_RELU, {PATTERN_TRANSDATA})
      .SetInputs(PATTERN_CAST, {PATTERN_RELU})
      .SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define ReluCastFusionPass pattern");
  return patterns;
}

Status ReluCastFusionPass::GetFusedNodes(const ge::ComputeGraph& graph, const Mapping& mapping,
                                         vector<ge::NodePtr>& fusedNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into GetFusedNodes");

  ge::NodePtr concatv2dNode = GetNodeFromMapping(PATTERN_CONCATV2D, mapping);
  FUSION_PASS_CHECK(concatv2dNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get ConcatV2D node"),
                    return NOT_CHANGED);

  ge::NodePtr transDataNode = GetNodeFromMapping(PATTERN_TRANSDATA, mapping);
  FUSION_PASS_CHECK(transDataNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get TransData node"),
                    return NOT_CHANGED);

  ge::NodePtr reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);
  FUSION_PASS_CHECK(reluNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get Relu node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reluNode->GetOutDataAnchor(0)->GetPeerInDataNodesSize() != 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Post node size of Relu should be 1"), return NOT_CHANGED);
  FUSION_PASS_CHECK(reluNode->GetOpDesc()->GetInputDesc(0).GetShape().IsUnknownShape(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Relu input is unknown shape"), return NOT_CHANGED);

  ge::NodePtr castNode = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(castNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get Cast node"),
                    return NOT_CHANGED);
  ge::OpDescPtr castDesc = castNode->GetOpDesc();
  DataType inputDataType = castDesc->GetInputDesc(0).GetDataType();
  FUSION_PASS_CHECK(inputDataType != DT_FLOAT && inputDataType != DT_FLOAT16,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Cast input type is not in {float16, float32}"),
                    return NOT_CHANGED);
  DataType outputDataType = castDesc->GetOutputDesc(0).GetDataType();
  FUSION_PASS_CHECK(outputDataType != DT_FLOAT && outputDataType != DT_FLOAT16,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Cast output type is not in {float16, float32}"),
                    return NOT_CHANGED);

  fusedNodes.push_back(concatv2dNode);
  fusedNodes.push_back(transDataNode);
  fusedNodes.push_back(reluNode);
  fusedNodes.push_back(castNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to GetFusedNodes");
  return SUCCESS;
}

Status ReluCastFusionPass::OptimizeNodesOrder(vector<ge::NodePtr>& fusedNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into OptimizeNodesOrder");

  ge::NodePtr transDataNode = fusedNodes[1];
  ge::NodePtr reluNode = fusedNodes[2];
  ge::NodePtr castNode = fusedNodes[3];

  // Step 1. Reconnect edges
  auto castInAnchor = castNode->GetInDataAnchor(0);
  castInAnchor->UnlinkAll();
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transDataNode->GetOutDataAnchor(0), castInAnchor) != ge::GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from TransData to Cast"),
      return FAILED);

  for (auto postInAnchor : castNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (postInAnchor != nullptr) {
      postInAnchor->UnlinkAll();
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(reluNode->GetOutDataAnchor(0), postInAnchor) != ge::GRAPH_SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from Relu to post node"),
          return FAILED);
    }
  }

  auto reluInAnchor = reluNode->GetInDataAnchor(0);
  reluInAnchor->UnlinkAll();
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(castNode->GetOutDataAnchor(0), reluInAnchor) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from Cast to Relu"),
                    return FAILED);

  // Step 2. Refresh data type
  ge::OpDescPtr castDesc = castNode->GetOpDesc();
  DataType outputDataType = castDesc->GetOutputDesc(0).GetDataType();

  ge::OpDescPtr reluDesc = reluNode->GetOpDesc();
  reluDesc->MutableInputDesc(0)->SetOriginDataType(outputDataType);
  reluDesc->MutableInputDesc(0)->SetDataType(outputDataType);
  reluDesc->MutableOutputDesc(0)->SetOriginDataType(outputDataType);
  reluDesc->MutableOutputDesc(0)->SetDataType(outputDataType);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to OptimizeNodesOrder");
  return SUCCESS;
}

Status ReluCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ReluCastFusionPass");

  vector<ge::NodePtr> fusedNodes;
  FUSION_PASS_CHECK(GetFusedNodes(graph, mapping, fusedNodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get fused nodes"), return NOT_CHANGED);

  Status optiRet = OptimizeNodesOrder(fusedNodes);
  FUSION_PASS_CHECK(optiRet != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to optimize order of nodes"),
                    return optiRet);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ReluCastFusionPass");
  return SUCCESS;
}

REGISTER_PASS("ReluCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, ReluCastFusionPass);
}  // namespace fe
