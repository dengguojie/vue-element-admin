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
 * \file reshape_transpose_fusion_pass.cpp
 * \brief confusionTranspose fusion pass(Transpose-Reshape --> confusionTranspose)
 */
#include "reshape_transpose_fusion_pass.h"

#include <iostream>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {

static const char PATTERN_TRANSPOSE[] = "FusedNodeTranspose";
static const char PATTERN_RESHAPE[] = "FusedNodeReshape";

vector<FusionPattern*> ReshapeTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReshapeTransposeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_TRANSPOSE, {"TransposeD"})
      .AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
      .SetInputs(PATTERN_TRANSPOSE, {PATTERN_RESHAPE})
      .SetOutput(PATTERN_TRANSPOSE);

  patterns.push_back(pattern);

  return patterns;
}

Status ReshapeTransposeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr transNode = GetNodeFromMapping(PATTERN_TRANSPOSE, mapping);
  FUSION_PASS_CHECK(transNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransposeNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr transDesc = transNode->GetOpDesc();
  FUSION_PASS_CHECK(transDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransposeNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::NodePtr reshapeNode = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  FUSION_PASS_CHECK(reshapeNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReshapeNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr reshapeDesc = reshapeNode->GetOpDesc();
  FUSION_PASS_CHECK(reshapeDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReshapeNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  if (reshapeNode->GetAllInDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 2 input anchor.");
    return NOT_CHANGED;
  }
  if (reshapeNode->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 1 output anchor.");
    return NOT_CHANGED;
  }
  if (reshapeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The TransposeD node should only have 1 output anchor that link to other nodes.");
    return NOT_CHANGED;
  }
  if (reshapeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != "TransposeD") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The TransposeD node should only have 1 output anchor that link to TransposeD.");
    return NOT_CHANGED;
  }
  if (transNode->GetAllInDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:[%s] only should have one input, but actually have %d.",
            transNode->GetName().c_str(), transNode->GetAllInDataAnchors().size());
    return NOT_CHANGED;
  }
  vector<int64_t> reshapeDimInfo = reshapeDesc->GetInputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> transposeDimInfo = transDesc->GetOutputDesc(0).GetOriginShape().GetDims();
  if (reshapeDimInfo.size() == 1) {
    if (PatternFusionUtil::IsUnknownShape(reshapeDimInfo[0])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (reshapeDimInfo[0] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s dimsize is [%zu], last one dimension should be divisible by 16, but actually is [%ld].",
              reshapeNode->GetName().c_str(), reshapeDimInfo.size(), reshapeDimInfo[0]);
      return NOT_CHANGED;
    }
  } else if (reshapeDimInfo.size() >= 2) {
    if (PatternFusionUtil::IsUnknownShape(reshapeDimInfo[reshapeDimInfo.size() - 1]) ||
        PatternFusionUtil::IsUnknownShape(reshapeDimInfo[reshapeDimInfo.size() - 2])) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (reshapeDimInfo[reshapeDimInfo.size() - 1] % 16 != 0 || reshapeDimInfo[reshapeDimInfo.size() - 2] % 16 != 0) {
      OP_LOGI(
          FUSED_OP_TYPE.c_str(),
          "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
          reshapeNode->GetName().c_str(), reshapeDimInfo.size(), reshapeDimInfo[reshapeDimInfo.size() - 2],
          reshapeDimInfo[reshapeDimInfo.size() - 1]);
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", reshapeNode->GetName().c_str(),
            reshapeDimInfo.size());
    return NOT_CHANGED;
  }
  if (transposeDimInfo.size() == 1) {
    if (PatternFusionUtil::IsUnknownShape(transposeDimInfo[0])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (transposeDimInfo[0] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s dimsize is [%zu], last one dimension should be divisible by 16, but actually is [%ld].",
              transNode->GetName().c_str(), transposeDimInfo.size(), transposeDimInfo[0]);
      return NOT_CHANGED;
    }
  } else if (transposeDimInfo.size() >= 2) {
    if (PatternFusionUtil::IsUnknownShape(transposeDimInfo[transposeDimInfo.size() - 1]) ||
        PatternFusionUtil::IsUnknownShape(transposeDimInfo[transposeDimInfo.size() - 2])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (transposeDimInfo[transposeDimInfo.size() - 1] % 16 != 0 ||
        transposeDimInfo[transposeDimInfo.size() - 2] % 16 != 0) {
      OP_LOGI(
          FUSED_OP_TYPE.c_str(),
          "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
          transNode->GetName().c_str(), transposeDimInfo.size(), transposeDimInfo[transposeDimInfo.size() - 2],
          transposeDimInfo[transposeDimInfo.size() - 1]);
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", transNode->GetName().c_str(),
            transposeDimInfo.size());
    return NOT_CHANGED;
  }
  vector<int32_t> permValue;
  ge::AttrUtils::GetListInt(transDesc, "perm", permValue);
  ge::AttrUtils::SetListInt(reshapeDesc, "perm", permValue);
  ge::AttrUtils::SetBool(reshapeDesc, "transpose_first", false);
  ge::GeTensorDesc reshapeOutputTensor = reshapeDesc->GetOutputDesc(0);
  ge::GeShape reshapeOutputShape = reshapeOutputTensor.GetShape();
  vector<int64_t> dimInfo = reshapeOutputShape.GetDims();
  reshapeDesc->SetName(reshapeDesc->GetName() + "/ConfusionTranspose");
  reshapeDesc->SetType("ConfusionTranspose");
  reshapeDesc->UpdateOutputDesc(0, transDesc->GetOutputDesc(0));
  if (transNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of transNode is [%d].",
            transNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr &inAnchorPtr : transNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(reshapeNode->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  transNode->GetName().c_str(), reshapeNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              transNode->GetName().c_str(), reshapeNode->GetName().c_str());
    }
  }
  transNode->GetOutDataAnchor(0)->UnlinkAll();
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(transNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node:[%s] failed.", transDesc->GetName().c_str()),
                    return FAILED);
  ge::NodePtr confusionTransposeD = nullptr;
  std::string fusionOpType = "ConfusionTransposeD";
  std::vector<PassAttrInfo> confusionTransposeAttrInfo;

  PassAttrInfo shape = {1, "shape", "SetListInt"};
  confusionTransposeAttrInfo.push_back(shape);
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, reshapeNode, fusionOpType, confusionTransposeAttrInfo,
                                                      confusionTransposeD);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Tile has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  ge::AttrUtils::SetListInt(confusionTransposeD->GetOpDesc(), "shape", dimInfo);
  fusionNodes.push_back(confusionTransposeD);
  
  return SUCCESS;
}
REGISTER_PASS("ReshapeTransposeFusionPass", BUILT_IN_GRAPH_PASS, ReshapeTransposeFusionPass);
}  // namespace fe
