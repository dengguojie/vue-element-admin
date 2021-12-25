/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file tabulatefusiongrad_fusion_pass.cc
 * \brief TabulateFusionGrad fusion pass(Parallel TabulateFusionGrad)
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
#include "deep_md_fusion_pass_util.h"
#include "tabulatefusiongrad_fusion_pass.h"

namespace fe {
static const std::string PATTERN_TABULATEFUSIONGRAD = "TabulateFusionGrad";
static const std::string OP_TYPE_TABULATEFUSIONGRAD = "TabulateFusionGrad";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        inputs                                  inputs
 *          |                                     /   \
 *  TabulateFusionGrad    ==>    TabulateFusionGrad   TabulateFusionGrad
 *          |                                   | \   / |
 *       outputs                                |  \ /  |
 *                                              |  / \  |
 *                                           Concat   Concat
 *                                                \   /
 *                                               outputs
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> TabulateFusionGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string passName = "TabulateFusionGradFusionPass";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_TABULATEFUSIONGRAD, {OP_TYPE_TABULATEFUSIONGRAD}).SetOutput(PATTERN_TABULATEFUSIONGRAD);
  patterns.push_back(pattern);
  return patterns;
}

Status TabulateFusionGradFusionPass::UpdateTabulateFusionGradNode(ge::ComputeGraph& graph, ge::NodePtr& tabulateNodeAic,
                                                                  ge::NodePtr& tabulateNodeVec) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into Update TabulateFusion node");
  ge::OpDescPtr tabulateAicDesc = tabulateNodeAic->GetOpDesc();
  ge::OpDescPtr tabulateVecDesc = tabulateNodeVec->GetOpDesc();
  FUSION_PASS_CHECK(
      tabulateAicDesc == nullptr || tabulateVecDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get Aic or Vec OpDesc of TabulateFusion"),
      return PARAM_INVALID);

  ge::GeTensorDesc dyDemXDescAic = tabulateAicDesc->GetOutputDesc(0);
  ge::GeTensorDesc dyDemXDescVec = tabulateVecDesc->GetOutputDesc(0);
  ge::GeShape dyDemXShapeAic = dyDemXDescAic.GetShape();
  ge::GeShape dyDemXShapeVec = dyDemXDescVec.GetShape();
  FUSION_PASS_CHECK(dyDemXShapeAic.GetDimNum() != 2 || dyDemXShapeVec.GetDimNum() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The output shape of dy_dem_x must be 2"),
                    return PARAM_INVALID);

  ge::GeTensorDesc dyDemDescAic = tabulateAicDesc->GetOutputDesc(1);
  ge::GeTensorDesc dyDemDescVec = tabulateVecDesc->GetOutputDesc(1);
  ge::GeShape dyDemShapeAic = dyDemDescAic.GetShape();
  ge::GeShape dyDemShapeVec = dyDemDescVec.GetShape();
  FUSION_PASS_CHECK(dyDemShapeAic.GetDimNum() != 3 || dyDemShapeVec.GetDimNum() != 3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The output shape of dy_dem must be 3"),
                    return PARAM_INVALID);

  if (dyDemXShapeAic.GetDim(0) > 0) {
    int64_t nloc = dyDemXShapeAic.GetDim(0);
    int64_t splitCount = 2;
    int64_t ceilValue = (nloc + splitCount - 1) / splitCount;

    dyDemXShapeAic.SetDim(0, ceilValue);
    dyDemXDescAic.SetShape(dyDemXShapeAic);
    dyDemXDescAic.SetOriginShape(dyDemXShapeAic);
    tabulateAicDesc->UpdateOutputDesc(0, dyDemXDescAic);

    dyDemXShapeVec.SetDim(0, nloc - ceilValue);
    dyDemXDescVec.SetShape(dyDemXShapeVec);
    dyDemXDescVec.SetOriginShape(dyDemXShapeVec);
    tabulateVecDesc->UpdateOutputDesc(0, dyDemXDescVec);

    dyDemShapeAic.SetDim(0, ceilValue);
    dyDemDescAic.SetShape(dyDemShapeAic);
    dyDemDescAic.SetOriginShape(dyDemShapeAic);
    tabulateAicDesc->UpdateOutputDesc(1, dyDemDescAic);

    dyDemShapeVec.SetDim(0, nloc - ceilValue);
    dyDemDescVec.SetShape(dyDemShapeVec);
    dyDemDescVec.SetOriginShape(dyDemShapeVec);
    tabulateVecDesc->UpdateOutputDesc(1, dyDemDescVec);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to Update TabulateFusion node");
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TabulateFusionGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into TabulateFusionGrad fusion pass");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_TABULATEFUSIONGRAD, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get TabulateFusionGrad Node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::CheckSplitInitInfo(FUSED_OP_TYPE, fusedNode) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to check split_count, split_index"), return NOT_CHANGED);

  ge::NodePtr nodeAic = nullptr;
  ge::NodePtr nodeVec = nullptr;
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::SplitNodeToAICoreAndVectorCore(FUSED_OP_TYPE, graph, fusedNode, nodeAic,
                                                                         nodeVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to split TabulateFusionGrad node"),
                    return FAILED);
  newNodes.push_back(nodeAic);
  newNodes.push_back(nodeVec);

  FUSION_PASS_CHECK(UpdateTabulateFusionGradNode(graph, nodeAic, nodeVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to update tabulate shape"),
                    return FAILED);

  int32_t concatDim = 0;
  int32_t concatN = 2;
  vector<int32_t> concatAttrs{concatDim, concatN};

  ge::NodePtr concatDyDemXNode = nullptr;
  std::string concatDyDemXNodeName = fusedNode->GetName() + "/Concat/dy_dem_x";
  vector<ge::NodePtr> tabulateNodes = {fusedNode, nodeAic, nodeVec};
  FUSION_PASS_CHECK(
      DeepMdFusionPassUtil::CreateConcatNodeAfterSplitNode(FUSED_OP_TYPE, graph, concatDyDemXNode, concatDyDemXNodeName,
                                                           tabulateNodes, 0, concatAttrs) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat(dy_dem_x) node"), return FAILED);
  newNodes.push_back(concatDyDemXNode);

  ge::NodePtr concatDyDemNode = nullptr;
  std::string concatDyDemNodeName = fusedNode->GetName() + "/Concat/dy_dem";
  FUSION_PASS_CHECK(
      DeepMdFusionPassUtil::CreateConcatNodeAfterSplitNode(FUSED_OP_TYPE, graph, concatDyDemNode, concatDyDemNodeName,
                                                           tabulateNodes, 1, concatAttrs) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat(dy_dem) node"), return FAILED);
  newNodes.push_back(concatDyDemNode);

  for (auto inAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatDyDemXNode->GetOutDataAnchor(0), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output dy_dem_x to output node"),
        return FAILED);
  }
  for (auto inAnchor : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatDyDemNode->GetOutDataAnchor(0), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output dy_dem_x to output node"),
        return FAILED);
  }

  FUSION_PASS_CHECK(DeepMdFusionPassUtil::ClearFusedNode(FUSED_OP_TYPE, graph, fusedNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to clear TabulateFusionGrad node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to TabulateFusionGrad fusion pass");
  return SUCCESS;
}
REGISTER_PASS("TabulateFusionGradFusionPass", BUILT_IN_GRAPH_PASS, TabulateFusionGradFusionPass);
}  // namespace fe
