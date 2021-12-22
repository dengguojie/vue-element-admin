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
 * \file fusedbatchnorm_bert_fusion_pass.cpp
 * \brief fusedbatchnormgrad_bert fusion pass
 */
#include "fusedbatchnorm_bert_fusion_pass.h"
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {

static const string PATTERN_FUSEDBATCHNORM = "BatchNorm";
static const string PASS_OP_TYPE_BATCHNORM = "BatchNorm";

vector<FusionPattern*> FusedBatchNormBertFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("FusedBatchNormBertFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDBATCHNORM, {PASS_OP_TYPE_BATCHNORM}).SetOutput(PATTERN_FUSEDBATCHNORM);
  patterns.push_back(pattern);
  return patterns;
}

Status FusedBatchNormBertFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr batchNormNode = GetNodeFromMapping(PATTERN_FUSEDBATCHNORM, mapping);
  FUSION_PASS_CHECK(batchNormNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "batchNormNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr batchNormDesc = batchNormNode->GetOpDesc();
  std::string batchNormName = batchNormDesc->GetName();
  // validation
  FUSION_PASS_CHECK(batchNormDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "batchNormNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  if (batchNormDesc->GetInputsSize() != 3) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] should have 3 input desc, but actually is %d", batchNormName.c_str(),
            batchNormDesc->GetInputsSize());
    return NOT_CHANGED;
  }
  if (batchNormDesc->GetOutputsSize() != 5) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] should have 6 output desc, but actually is %d", batchNormName.c_str(),
            batchNormDesc->GetOutputsSize());
    return NOT_CHANGED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has %u output anchor.", batchNormDesc->GetName().c_str(),
          batchNormNode->GetAllOutDataAnchors().size());
  if (!(batchNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().empty()) ||
      !(batchNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().empty())) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s 1st and 2nd output anchor shouldn't link to any other input anchors.",
            batchNormName.c_str());
    return NOT_CHANGED;
  }
  if (batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty() ||
      batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().empty() ||
      batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s 0/3/4 output should link to other node's input.", batchNormName.c_str());
    return NOT_CHANGED;
  }
  // end validation

  // reduce desc
  ge::OpDescPtr batchNormReduceDesc = AttrUtils::CloneOpDesc(batchNormDesc);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s input size is %d, all size is %d.", batchNormName.c_str(),
          batchNormReduceDesc->GetInputsSize(), batchNormReduceDesc->GetAllInputsSize());
  FUSION_PASS_CHECK(batchNormReduceDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                   batchNormName.c_str()),
                    return PARAM_INVALID);
  batchNormReduceDesc->SetName(batchNormDesc->GetName() + "/BNTrainingReduce");
  batchNormReduceDesc->SetType("BNTrainingReduce");
  if (batchNormReduceDesc->GetInputsSize() < 3) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of input desc", batchNormReduceDesc->GetInputsSize());
    return NOT_CHANGED;
  }
  if (batchNormReduceDesc->GetOutputsSize() < 5) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of output desc",
            batchNormReduceDesc->GetOutputsSize());
    return NOT_CHANGED;
  }
  OpDescUtils::ClearInputDesc(batchNormReduceDesc, 4);
  OpDescUtils::ClearInputDesc(batchNormReduceDesc, 3);
  OpDescUtils::ClearInputDesc(batchNormReduceDesc, 2);
  OpDescUtils::ClearInputDesc(batchNormReduceDesc, 1);
  for (int index = 4; index >= 0; index--) {
    OpDescUtils::ClearOutputDesc(batchNormReduceDesc, index);
  }
  batchNormReduceDesc->AddOutputDesc("sum", batchNormDesc->GetInputDesc(1));
  batchNormReduceDesc->AddOutputDesc("square_sum", batchNormDesc->GetInputDesc(2));
  // end reduce desc
  // update desc
  ge::OpDescPtr batchNormUpdateV2Desc = AttrUtils::CloneOpDesc(batchNormDesc);
  FUSION_PASS_CHECK(batchNormUpdateV2Desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                   batchNormName.c_str()),
                    return PARAM_INVALID);
  batchNormUpdateV2Desc->SetName(batchNormDesc->GetName() + "/BNTrainingUpdateV2");
  batchNormUpdateV2Desc->SetType("BNTrainingUpdateV2");
  if (batchNormUpdateV2Desc->GetOutputsSize() < 5) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of output desc",
            batchNormUpdateV2Desc->GetOutputsSize());
    return NOT_CHANGED;
  }
  if (batchNormUpdateV2Desc->GetInputsSize() < 3) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of input desc",
            batchNormUpdateV2Desc->GetInputsSize());
    return NOT_CHANGED;
  }
  batchNormUpdateV2Desc->AddInputDesc(3, batchNormDesc->GetInputDesc(1));
  batchNormUpdateV2Desc->AddInputDesc(4, batchNormDesc->GetInputDesc(2));
  OpDescUtils::ClearOutputDesc(batchNormUpdateV2Desc, 2);
  OpDescUtils::ClearOutputDesc(batchNormUpdateV2Desc, 1);
  // end update desc

  // add reduce and update node to graph
  ge::NodePtr batchNormReduceNode = graph.AddNode(batchNormReduceDesc);
  ge::NodePtr batchNormUpdateV2Node = graph.AddNode(batchNormUpdateV2Desc);
  FUSION_PASS_CHECK(
      batchNormReduceNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                     batchNormReduceNode->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      batchNormUpdateV2Node == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                     batchNormUpdateV2Node->GetName().c_str()),
      return PARAM_INVALID);

  // update input and output name map
  FUSION_PASS_CHECK(PatternFusionUtil::UpdateInputAndOutputName(batchNormReduceDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Update fusionNode:%s input and output name failed.",
                            batchNormReduceDesc->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(PatternFusionUtil::UpdateInputAndOutputName(batchNormUpdateV2Desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Update fusionNode:%s input and output name failed.",
                            batchNormUpdateV2Desc->GetName().c_str()), return FAILED);
  // connect edge for reduce node
  // input date anchor
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                   batchNormReduceNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index failed.",
              batchNormName.c_str(), batchNormReduceNode->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
          batchNormName.c_str(), batchNormReduceNode->GetName().c_str());
  // output data anchor
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS !=
          ge::GraphUtils::AddEdge(batchNormReduceNode->GetOutDataAnchor(0), batchNormUpdateV2Node->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 1st index to fusion node:%s's 2nd index failed.",
              batchNormReduceNode->GetName().c_str(), batchNormUpdateV2Node->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 2nd index.",
          batchNormReduceNode->GetName().c_str(), batchNormUpdateV2Node->GetName().c_str());
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS !=
          ge::GraphUtils::AddEdge(batchNormReduceNode->GetOutDataAnchor(1), batchNormUpdateV2Node->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 2nd index to fusion node:%s's 3rd index failed.",
              batchNormReduceNode->GetName().c_str(), batchNormUpdateV2Node->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 2nd index to fusion node:%s's 3rd index.",
          batchNormReduceNode->GetName().c_str(), batchNormUpdateV2Node->GetName().c_str());
  // input control anchor
  for (unsigned int i = 0; i < batchNormNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        ge::GRAPH_SUCCESS !=
            ge::GraphUtils::AddEdge(batchNormNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                    batchNormReduceNode->GetInControlAnchor()),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                batchNormName.c_str(), i, batchNormReduceNode->GetName().c_str()),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            batchNormName.c_str(), i, batchNormReduceNode->GetName().c_str());
  }

  // connect edge for update node
  // input data anchor
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                   batchNormUpdateV2Node->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index failed.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
          batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                   batchNormUpdateV2Node->GetInDataAnchor(3)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 1st index to fusion node:%s's 4th index failed.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 4th index.",
          batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                   batchNormUpdateV2Node->GetInDataAnchor(4)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's 1st index to fusion node:%s's 5th index failed.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 5th index.",
          batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());

  // output data anchor
  if (batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of batchNormNode is [%d].",
            batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(batchNormUpdateV2Node->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());
    }
  }

  if (batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of batchNormNode is [%d].",
            batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(batchNormUpdateV2Node->GetOutDataAnchor(1), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "Add edge from fused node:%s's index to fusion node:%s's 2nd index failed.",
                  batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 2nd index.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());
    }
  }
  if (batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of batchNormNode is [%d].",
            batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(batchNormUpdateV2Node->GetOutDataAnchor(2), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "Add edge from fused node:%s's index to fusion node:%s's 3rd index failed.",
                  batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 3rd index.",
              batchNormName.c_str(), batchNormUpdateV2Node->GetName().c_str());
    }
  }

  // input control anchor
  for (unsigned int i = 0; i < batchNormNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        ge::GRAPH_SUCCESS !=
            ge::GraphUtils::AddEdge(batchNormNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                    batchNormUpdateV2Node->GetInControlAnchor()),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                batchNormName.c_str(), i, batchNormUpdateV2Node->GetName().c_str()),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            batchNormName.c_str(), i, batchNormUpdateV2Node->GetName().c_str());
  }

  // remove edge for batchnorm node
  if (batchNormNode->GetInControlAnchor() != nullptr) {
    batchNormNode->GetInControlAnchor()->UnlinkAll();
  }
  for (auto inAnchor : batchNormNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(batchNormNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove batchNormNode node[%s] failed",
                                                   batchNormName.c_str()),
                    return FAILED);
  fusionNodes.push_back(batchNormReduceNode);
  fusionNodes.push_back(batchNormUpdateV2Node);
  return SUCCESS;
}
REGISTER_PASS("FusedBatchNormBertFusionPass", BUILT_IN_GRAPH_PASS, FusedBatchNormBertFusionPass);
}  // namespace fe
