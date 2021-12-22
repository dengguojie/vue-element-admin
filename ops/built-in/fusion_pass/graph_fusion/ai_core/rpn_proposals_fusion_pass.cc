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
 * \file rpn_proposals_fusion_pass.cc
 * \brief RpnProposals fusion pass
 *   (RpnProposalsD --> ScoreFilterPreSort & RpnProposalsPostProcessing)
 */
#include "rpn_proposals_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

static const int64_t C0 = 6002 * 8;
static const int64_t C1 = 8;
static const int64_t C2 = 8;

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "RpnProposalsD";
static const std::string PATTERN_FUSEDNODE = "RpnProposals";

vector<FusionPattern*> RpnProposalsFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("RpnProposalsFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "RpnProposalsFusionPass DefinePatterns add pattern RpnProposals.");

  return patterns;
}

Status RpnProposalsFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // scoreFilterPreSort
  // rpnProposalPostProcessing
  ge::OpDescPtr scoreFilterPreSortDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      scoreFilterPreSortDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                     fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  ge::OpDescPtr rpnProposalPostProcessingDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      rpnProposalPostProcessingDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                     fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  scoreFilterPreSortDesc->SetName(fusedDesc->GetName() + "/ScoreFilterPreSort");
  rpnProposalPostProcessingDesc->SetName(fusedDesc->GetName() + "/RpnProposalPostProcessing");
  scoreFilterPreSortDesc->SetType("ScoreFilterPreSort");
  rpnProposalPostProcessingDesc->SetType("RpnProposalPostProcessing");

  if (rpnProposalPostProcessingDesc->GetInputsSize() != 2) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of input desc",
            rpnProposalPostProcessingDesc->GetInputsSize());
    return NOT_CHANGED;
  }

  if (scoreFilterPreSortDesc->GetOutputsSize() != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%d] of output desc",
            scoreFilterPreSortDesc->GetOutputsSize());
    return NOT_CHANGED;
  }

  OpDescUtils::ClearOutputDesc(scoreFilterPreSortDesc, 0);
  vector<int64_t> output0Shape;
  output0Shape.push_back(C0);
  output0Shape.push_back(C1);
  ge::GeTensorDesc output_tensor0Desc;
  output_tensor0Desc.SetShape(ge::GeShape(output0Shape));
  output_tensor0Desc.SetFormat(ge::FORMAT_ND);  // ND
  output_tensor0Desc.SetDataType(ge::DT_FLOAT16);
  output_tensor0Desc.SetOriginShape(ge::GeShape(output0Shape));  // 6000x8x8
  output_tensor0Desc.SetOriginFormat(ge::FORMAT_ND);
  output_tensor0Desc.SetOriginDataType(ge::DT_FLOAT16);
  scoreFilterPreSortDesc->AddOutputDesc("x", output_tensor0Desc);
  vector<int64_t> output1Shape;
  output1Shape.push_back(C1);
  output1Shape.push_back(C2);
  ge::GeTensorDesc output_tensor1Desc;
  output_tensor1Desc.SetShape(ge::GeShape(output1Shape));
  output_tensor1Desc.SetFormat(ge::FORMAT_ND);  // ND
  output_tensor1Desc.SetDataType(ge::DT_UINT32);
  output_tensor1Desc.SetOriginShape(ge::GeShape(output1Shape));  // 8x8
  output_tensor1Desc.SetOriginFormat(ge::FORMAT_ND);
  output_tensor1Desc.SetOriginDataType(ge::DT_UINT32);
  scoreFilterPreSortDesc->AddOutputDesc("y", output_tensor1Desc);

  OpDescUtils::ClearInputDesc(rpnProposalPostProcessingDesc, 1);
  OpDescUtils::ClearInputDesc(rpnProposalPostProcessingDesc, 0);
  vector<int64_t> input0Shape;
  input0Shape.push_back(C0);
  input0Shape.push_back(C1);
  ge::GeTensorDesc input_tensor0Desc;
  input_tensor0Desc.SetShape(ge::GeShape(input0Shape));
  input_tensor0Desc.SetFormat(ge::FORMAT_ND);  // ND
  input_tensor0Desc.SetDataType(ge::DT_FLOAT16);
  input_tensor0Desc.SetOriginShape(ge::GeShape(input0Shape));  // 6000x8x8
  input_tensor0Desc.SetOriginFormat(ge::FORMAT_ND);
  input_tensor0Desc.SetOriginDataType(ge::DT_FLOAT16);
  rpnProposalPostProcessingDesc->AddInputDesc("x", input_tensor0Desc);
  vector<int64_t> input1Shape;
  input1Shape.push_back(C1);
  input1Shape.push_back(C2);
  ge::GeTensorDesc input_tensor1Desc;
  input_tensor1Desc.SetShape(ge::GeShape(input1Shape));
  input_tensor1Desc.SetFormat(ge::FORMAT_ND);  // ND
  input_tensor1Desc.SetDataType(ge::DT_UINT32);
  input_tensor1Desc.SetOriginShape(ge::GeShape(input1Shape));  // 8x8
  input_tensor1Desc.SetOriginFormat(ge::FORMAT_ND);
  input_tensor1Desc.SetOriginDataType(ge::DT_UINT32);
  rpnProposalPostProcessingDesc->AddInputDesc("y", input_tensor1Desc);

  // change attr
  FUSION_PASS_CHECK(SUCCESS != scoreFilterPreSortDesc->DelAttr("img_size"),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Delete the attr of img_size from scoreFilterPreSortDesc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(SUCCESS != scoreFilterPreSortDesc->DelAttr("min_size"),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Delete the attr of min_size from scoreFilterPreSortDesc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(
      SUCCESS != scoreFilterPreSortDesc->DelAttr("nms_threshold"),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Delete the attr of nms_threshold from scoreFilterPreSortDesc failed."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      SUCCESS != scoreFilterPreSortDesc->DelAttr("post_nms_num"),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Delete the attr of post_nms_num from scoreFilterPreSortDesc failed."),
      return PARAM_INVALID);

  ge::NodePtr scoreFilterPreSortNode = graph.AddNode(scoreFilterPreSortDesc);
  ge::NodePtr rpnProposalPostProcessingNode = graph.AddNode(rpnProposalPostProcessingDesc);
  FUSION_PASS_CHECK(scoreFilterPreSortNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            scoreFilterPreSortNode->GetName().c_str()),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rpnProposalPostProcessingNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            rpnProposalPostProcessingNode->GetName().c_str()),
                    return PARAM_INVALID);
  for (unsigned int i = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                           scoreFilterPreSortNode->GetInDataAnchor(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fusedNode->GetName().c_str(), i, scoreFilterPreSortNode->GetName().c_str(), i),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
            fusedNode->GetName().c_str(), i, scoreFilterPreSortNode->GetName().c_str(), i);
  }
  for (unsigned int i = 0; i < fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                           scoreFilterPreSortNode->GetInControlAnchor()),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                fusedNode->GetName().c_str(), i, scoreFilterPreSortNode->GetName().c_str()),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), i, scoreFilterPreSortNode->GetName().c_str());
  }

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(scoreFilterPreSortNode->GetOutDataAnchor(0),
                                         rpnProposalPostProcessingNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "Add edge from fused node:%s's index to fusion node:%s's 1st index failed. %lu %lu",
              scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str(),
              scoreFilterPreSortDesc->GetOutputsSize(), rpnProposalPostProcessingDesc->GetInputsSize()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
          scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str());

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(scoreFilterPreSortNode->GetOutDataAnchor(1),
                                         rpnProposalPostProcessingNode->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "Add edge from fused node:%s's index to fusion node:%s's 2nd index failed. %lu %lu",
              scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str(),
              scoreFilterPreSortDesc->GetOutputsSize(), rpnProposalPostProcessingDesc->GetInputsSize()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 2nd index.",
          scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str());

  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layerNormXNode is [%d].",
            fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      if (inAnchorPtr != nullptr) {
        inAnchorPtr->UnlinkAll();
      }
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(rpnProposalPostProcessingNode->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st index failed.",
                  fusedNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion node:%s .", fusedNode->GetName().c_str(),
              rpnProposalPostProcessingNode->GetName().c_str());
    }
  }

  // unlink scoreFilterPreSortNode out
  if (scoreFilterPreSortNode->GetOutControlAnchor() != nullptr) {
    scoreFilterPreSortNode->GetOutControlAnchor()->UnlinkAll();
  }
  // add control edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(scoreFilterPreSortNode->GetOutControlAnchor(),
                                         rpnProposalPostProcessingNode->GetInControlAnchor()),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from node:%s's control to node:%s's control failed.",
              scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str()),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from node:%s's control to node:%s's control index.",
          scoreFilterPreSortNode->GetName().c_str(), rpnProposalPostProcessingNode->GetName().c_str());
  for (unsigned int i = 0; i < fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(rpnProposalPostProcessingNode->GetOutControlAnchor(),
                                           fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().at(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "Add out control edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                fusedNode->GetName().c_str(), i, rpnProposalPostProcessingNode->GetName().c_str()),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "Add out control edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), i, rpnProposalPostProcessingNode->GetName().c_str());
  }

  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                                   fusedNode->GetName().c_str()),
                    return FAILED);
  fusionNodes.push_back(scoreFilterPreSortNode);
  fusionNodes.push_back(rpnProposalPostProcessingNode);
  return SUCCESS;
}

REGISTER_PASS("RpnProposalsFusionPass", BUILT_IN_GRAPH_PASS, RpnProposalsFusionPass);
}  // namespace fe
