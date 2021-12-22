/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file pass_through_second_fusion_pass.cpp
 * \brief
 */
#include "pass_through_second_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PATTERN_PASSTHROUGH[] = "PassThroughSecondFusion";
static const char PASS_THROUGH_NODE[] = "PassThrough";
static const char SPACE_TO_DEPTH_NODE[] = "SpaceToDepth";

Status PassThroughSecondFusionPass::SetNodeAttrAndOpDesc(ge::OpDescPtr& curOpDesc, const ge::OpDescPtr& oriOpDesc) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter PassThrough Second FusionPass Set NewNode Attr and OpDesc");

  int64_t stride = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(oriOpDesc, "stride", stride),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Can not get passthrough stride attr failed."),
                                                   return FAILED);
  ge::AttrUtils::SetInt(curOpDesc, "block_size", stride);

  ge::GeTensorDesc oriInputDesc = oriOpDesc->GetInputDesc(0);
  curOpDesc->AddInputDesc("x", oriInputDesc);

  ge::GeTensorDesc oriInput1Desc = oriOpDesc->GetInputDesc(1);
  curOpDesc->AddInputDesc("filter", oriInput1Desc);

  ge::GeTensorDesc oriOutputDesc = oriOpDesc->GetOutputDesc(0);
  curOpDesc->AddOutputDesc("y", oriOutputDesc);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass Set NewNode Attr and OpDesc success");
  return SUCCESS;
}

Status PassThroughSecondFusionPass::UnlinkEdge(ge::NodePtr oriNode) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter PassThrough Second FusionPass unlink ori node edge");

  for (auto inAnchor : oriNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  if (oriNode->GetInControlAnchor() != nullptr) {
    oriNode->GetInControlAnchor()->UnlinkAll();
  }
  for (auto outAnchor : oriNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (oriNode->GetOutControlAnchor() != nullptr) {
    oriNode->GetOutControlAnchor()->UnlinkAll();
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass unlink ori node edge success");
  return SUCCESS;
}

Status PassThroughSecondFusionPass::InsertCurNode(ge::ComputeGraph& graph, ge::NodePtr oriNode) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter PassThrough Second FusionPass insert new node");
  // Get Input Node
  ge::InDataAnchorPtr oriInAnchorPtr0 = oriNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputNode = oriBottomPeerAnchorPtr0->GetOwnerNode();
  ge::InDataAnchorPtr oriInAnchorPtr1 = oriNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr1 = oriInAnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr inputNode1 = oriBottomPeerAnchorPtr1->GetOwnerNode();
  // Get Output Node
  ge::OutDataAnchorPtr oriOutAnchorPtr0 = oriNode->GetOutDataAnchor(0);
  auto oriTopPeerAnchors = oriOutAnchorPtr0->GetPeerInDataAnchors();

  ge::OpDescPtr oriOpDesc = oriNode->GetOpDesc();

  FUSION_PASS_CHECK(SUCCESS != UnlinkEdge(oriNode), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Unlink OriNode Edge failed."),
                    return FAILED);

  ge::NodePtr curNode;
  ge::OpDescPtr curOpDesc;
  string curNodeName = "space2depth";
  const char* curNodeType = SPACE_TO_DEPTH_NODE;

  FUSION_PASS_MAKE_SHARED((curOpDesc = std::make_shared<ge::OpDesc>(oriNode->GetName(), curNodeType)),
                          return INTERNAL_ERROR);

  FUSION_PASS_CHECK(SUCCESS != SetNodeAttrAndOpDesc(curOpDesc, oriOpDesc),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set Node[%s] Attr and OpDesc failed.",
                                                   curNodeName.c_str()),
                    return FAILED);

  curNode = graph.AddNode(curOpDesc);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, curNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge from src node[%s] to dst node[%s] anchor 0 failed.",
                            inputNode->GetName().c_str(), curNodeName.c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr1, curNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge from src node[%s] to dst node[%s] anchor 1 failed.",
                            inputNode1->GetName().c_str(), curNodeName.c_str()),
                    return FAILED);

  for (uint64_t i = 0; i < oriTopPeerAnchors.size(); i++) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(curNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "add edge from src node[%s] to dst node[%s] failed.",
                              curNodeName.c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass insert new node success");
  return SUCCESS;
}

Status PassThroughSecondFusionPass::RemoveThisNode(ge::ComputeGraph& graph, ge::NodePtr thisNode) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter PassThrough Second FusionPass remove ori node");

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(thisNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove origin node[%s] node failed",
                                                   thisNode->GetName().c_str()),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass remove ori node success");
  return SUCCESS;
}

Status PassThroughSecondFusionPass::RemoveWeightNode(ge::ComputeGraph& graph, ge::NodePtr oriNode) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter PassThrough Second FusionPass remove weight node");
  ge::InDataAnchorPtr oriBottomInAnchorPtr = oriNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr = oriBottomInAnchorPtr->GetPeerOutAnchor();
  ge::NodePtr curNode = oriBottomPeerAnchorPtr->GetOwnerNode();
  string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(curNode);
  while (nodeType != "Constant" && nodeType != "Const") {
    ge::InDataAnchorPtr nodeInAnchorPtr = curNode->GetInDataAnchor(0);
    ge::OutDataAnchorPtr nodePeerAnchorPtr = nodeInAnchorPtr->GetPeerOutAnchor();
    ge::NodePtr bottomNode = nodePeerAnchorPtr->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != UnlinkEdge(curNode), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Unlink bottomNode Edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != RemoveThisNode(graph, curNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove bottomNode failed."),
                                                     return FAILED);
    curNode = bottomNode;
  }
  FUSION_PASS_CHECK(SUCCESS != UnlinkEdge(curNode), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Unlink Weight Node Edge failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != RemoveThisNode(graph, curNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Weight Node failed."), return FAILED);

  oriBottomInAnchorPtr->UnlinkAll();

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass remove weight node success");
  return SUCCESS;
}

vector<FusionPattern*> PassThroughSecondFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PassThroughSecondPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_PASSTHROUGH, {PASS_THROUGH_NODE}).SetOutput(PATTERN_PASSTHROUGH);

  patterns.push_back(pattern);

  return patterns;
}

Status PassThroughSecondFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into PassThrough Second FusionPass");
  // diag node
  ge::NodePtr passThroughNode = GetNodeFromMapping(PATTERN_PASSTHROUGH, mapping);
  FUSION_PASS_CHECK(passThroughNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "passThroughNode is null, fusion failed."),
                                                   return PARAM_INVALID);
  ge::OpDescPtr passThroughDesc = passThroughNode->GetOpDesc();
  FUSION_PASS_CHECK(passThroughDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "passThroughDesc is null, fusion failed."),
                                                   return PARAM_INVALID);

  FUSION_PASS_CHECK(passThroughDesc->GetInputsSize() == 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Passthrough ops not changed, dot not fusion."), return NOT_CHANGED);

  if (passThroughDesc->GetInputDesc(0).GetDataType() != DT_FLOAT16) {
    FUSION_PASS_CHECK(SUCCESS != RemoveWeightNode(graph, passThroughNode),
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove WeightNode failed."), return NOT_CHANGED);
  } else {
    FUSION_PASS_CHECK(SUCCESS != InsertCurNode(graph, passThroughNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Insert CurNode failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != RemoveThisNode(graph, passThroughNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove OriNode failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "PassThrough Second FusionPass success");
  return SUCCESS;
}

REGISTER_PASS("PassThroughSecondFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, PassThroughSecondFusionPass);
}  // namespace fe
