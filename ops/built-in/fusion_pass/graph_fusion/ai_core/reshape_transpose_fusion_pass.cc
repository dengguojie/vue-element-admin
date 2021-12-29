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
const int32_t CLEAR_OUTPUT_INDEX_TWO = 2;
const int32_t RESHAPE_DIM_UNIT = 16;
static const char PATTERN_TRANSPOSE[] = "FusedNodeTranspose";
static const char PATTERN_RESHAPE[] = "FusedNodeReshape";
static const string FUSION_OP_TYPE = "ConfusionTransposeD";
static const string OP_TRANSPOSE_D = "TransposeD";
static const string OP_RESHAPE = "Reshape";

vector<FusionPattern*> ReshapeTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReshapeTransposeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_TRANSPOSE, {OP_TRANSPOSE_D})
      .AddOpDesc(PATTERN_RESHAPE, {OP_RESHAPE})
      .SetInputs(PATTERN_TRANSPOSE, {PATTERN_RESHAPE})
      .SetOutput(PATTERN_TRANSPOSE);

  patterns.push_back(pattern);

  return patterns;
}

bool ReshapeTransposeFusionPass::CheckTransposeInfo(const ge::NodePtr nodePtr) {
  FUSION_PASS_CHECK(nodePtr->GetAllInDataAnchors().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Node:[%s] only should have one input, but actually have %d.", nodePtr->GetName().c_str(),
                            nodePtr->GetAllInDataAnchors().size()),
                    return false);

  ge::OpDescPtr opDesc = nodePtr->GetOpDesc();
  FUSION_PASS_CHECK(opDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "TransposeNode's op desc is null, fusion failed."),
                    return false);

  vector<int64_t> transposeDimInfo = opDesc->GetOutputDesc(0).GetOriginShape().GetDims();
  if (transposeDimInfo.size() == 1) {
    FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(transposeDimInfo[0]),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape."),
                      return false);
    FUSION_PASS_CHECK(transposeDimInfo[0] % 16 != 0,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], last one dimension should be"
                              "divisible by 16, but actually is [%ld].", nodePtr->GetName().c_str(),
                              transposeDimInfo.size(), transposeDimInfo[0]),
                      return false);
  } else if (transposeDimInfo.size() >= CLEAR_OUTPUT_INDEX_TWO) {
    FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(transposeDimInfo[transposeDimInfo.size() - 1]) ||
                        PatternFusionUtil::IsUnknownShape(transposeDimInfo[transposeDimInfo.size() - 2]),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape."),
                      return false);
    FUSION_PASS_CHECK(transposeDimInfo[transposeDimInfo.size() - 1] % 16 != 0 ||
                        transposeDimInfo[transposeDimInfo.size() - 2] % 16 != 0,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                              "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16,"
                              "but actually is [%ld] and [%ld].",
                              nodePtr->GetName().c_str(),
                              transposeDimInfo.size(), transposeDimInfo[transposeDimInfo.size() - 2],
                              transposeDimInfo[transposeDimInfo.size() - 1]),
                      return false);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", nodePtr->GetName().c_str(),
            transposeDimInfo.size());
    return false;
  }
  return true;
}

bool ReshapeTransposeFusionPass::ReLinkEdge(const ge::NodePtr& removeNode, const ge::NodePtr mainNode) const {
  if (removeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of removeNode is [%d].",
            removeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr &inAnchorPtr : removeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(mainNode->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index to fusion"
                                                                "node:%s's 1st index failed.",
                                         removeNode->GetName().c_str(), mainNode->GetName().c_str()),
          return false);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              removeNode->GetName().c_str(), mainNode->GetName().c_str());
    }
  }
  removeNode->GetOutDataAnchor(0)->UnlinkAll();
  return true;
}

Status ReshapeTransposeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Start to ReshapeTransposeFusionPass.");
  ge::NodePtr transNode = GetNodeFromMapping(PATTERN_TRANSPOSE, mapping);
  FUSION_PASS_CHECK(transNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "TransposeNode is null, fusion failed."), return PARAM_INVALID);
  ge::NodePtr reshapeNode = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  FUSION_PASS_CHECK(reshapeNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "ReshapeNode is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr reshapeDesc = reshapeNode->GetOpDesc();
  FUSION_PASS_CHECK(reshapeDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "ReshapeNode's OpDesc is null, fusion failed."), return PARAM_INVALID);
  if (reshapeNode->GetAllInDataAnchors().size() != CLEAR_OUTPUT_INDEX_TWO) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 2 input anchor.");
    return NOT_CHANGED;
  }
  if (reshapeNode->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 1 output anchor.");
    return NOT_CHANGED;
  }

  bool is_multi_trans = false;
  vector<ge::NodePtr> transposeNodes;
  for (auto anchor : reshapeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(anchor == nullptr || anchor->GetOwnerNode() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "there is a nullptr in reshape node's indata anchors."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(anchor->GetOwnerNode()->GetType() != OP_TRANSPOSE_D,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                              "reshape node link a node that is not TransposeD."),
                      return NOT_CHANGED);
    transposeNodes.push_back(anchor->GetOwnerNode());
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Get Transpose node num [%d].", transposeNodes.size());
  FUSION_PASS_CHECK(transposeNodes.size() < 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "The Reshape node should only have 1 output anchor that link to TransposeD."),
                    return NOT_CHANGED);
  is_multi_trans = transposeNodes.size() > 1;

  vector<int64_t> reshapeDimInfo = reshapeDesc->GetInputDesc(0).GetOriginShape().GetDims();
  if (reshapeDimInfo.size() == 1) {
    if (PatternFusionUtil::IsUnknownShape(reshapeDimInfo[0])) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (reshapeDimInfo[0] % RESHAPE_DIM_UNIT != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s dimsize is [%zu], last one dimension should be divisible by 16, but actually is [%ld].",
              reshapeNode->GetName().c_str(), reshapeDimInfo.size(), reshapeDimInfo[0]);
      return NOT_CHANGED;
    }
  } else if (reshapeDimInfo.size() >= CLEAR_OUTPUT_INDEX_TWO) {
    if (PatternFusionUtil::IsUnknownShape(reshapeDimInfo[reshapeDimInfo.size() - 1]) ||
        PatternFusionUtil::IsUnknownShape(reshapeDimInfo[reshapeDimInfo.size() - CLEAR_OUTPUT_INDEX_TWO])) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ReshapeTransposeFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (reshapeDimInfo[reshapeDimInfo.size() - 1] % RESHAPE_DIM_UNIT != 0 || reshapeDimInfo[
        reshapeDimInfo.size() - 2] % RESHAPE_DIM_UNIT != 0) {
      OP_LOGI(
          FUSED_OP_TYPE.c_str(),
          "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
          reshapeNode->GetName().c_str(), reshapeDimInfo.size(), reshapeDimInfo[reshapeDimInfo.size() -
          CLEAR_OUTPUT_INDEX_TWO], reshapeDimInfo[reshapeDimInfo.size() - 1]);
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", reshapeNode->GetName().c_str(),
            reshapeDimInfo.size());
    return NOT_CHANGED;
  }

  for (auto node : transposeNodes) {
    FUSION_PASS_CHECK(!CheckTransposeInfo(node),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Transpose node info is invalid, fusion fail."),
                      return NOT_CHANGED);
  }

  auto transDesc = transposeNodes.at(0)->GetOpDesc();
  vector<int64_t> transposeShapeVec = transDesc->GetOutputDesc(0).GetOriginShape().GetDims();
  vector<int32_t> permValue;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(transDesc, "perm", permValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "TransposeD nodes' perm value attrs are different."),
                    return PARAM_INVALID);

  if (is_multi_trans) {
    for (size_t i = 1; i < transposeNodes.size(); i++) {
      auto nodeDesc = transposeNodes.at(i)->GetOpDesc();
      vector<int32_t> tmpPerm;
      FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(nodeDesc, "perm", tmpPerm),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "TransposeD nodes' perm value attrs are different."),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(tmpPerm != permValue,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "TransposeD nodes' perm value attrs are different."),
                        return NOT_CHANGED);
      vector<int64_t> tmpShapeVec = nodeDesc->GetOutputDesc(0).GetOriginShape().GetDims();
      FUSION_PASS_CHECK(tmpShapeVec != transposeShapeVec,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "TransposeD nodes' output shapes are different."),
                        return NOT_CHANGED);
    }
  }

  ge::AttrUtils::SetListInt(reshapeDesc, "perm", permValue);
  ge::AttrUtils::SetBool(reshapeDesc, "transpose_first", false);
  ge::GeTensorDesc reshapeOutputTensor = reshapeDesc->GetOutputDesc(0);
  ge::GeShape reshapeOutputShape = reshapeOutputTensor.GetShape();
  vector<int64_t> dimInfo = reshapeOutputShape.GetDims();
  reshapeDesc->SetName(reshapeDesc->GetName() + "/" + FUSED_OP_TYPE);
  reshapeDesc->SetType(FUSED_OP_TYPE);
  reshapeDesc->UpdateOutputDesc(0, transDesc->GetOutputDesc(0));

  for (auto node : transposeNodes) {
    FUSION_PASS_CHECK(!ReLinkEdge(node, reshapeNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Reshape relink edge with transpose[%s] node fail.",
                                                     node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Remove node:[%s] failed.", node->GetName().c_str()),
                      return FAILED);
  }

  ge::NodePtr confusionTransposeD = nullptr;
  std::vector<PassAttrInfo> confusionTransposeAttrInfo;

  PassAttrInfo shape = {1, "shape", "SetListInt"};
  confusionTransposeAttrInfo.push_back(shape);
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, reshapeNode, FUSION_OP_TYPE, confusionTransposeAttrInfo,
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
