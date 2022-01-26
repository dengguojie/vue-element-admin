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
 * \file padd_update_fusion_pass.cc
 * \brief pad_d update fusion pass(pad_d --> pad)
 */
#include "padd_update_fusion_pass.h"

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
static const std::string PATTERN_PADD = "PadD";
static const std::string OP_TYPE_PAD = "Pad";
static const std::string OP_TYPE_PADD = "PadD";
static const std::string PADDINGS = "paddings";

vector<FusionPattern*> PaddUpdateFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into define PaddUpdateFusionPass pattern");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PaddUpdateFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {OP_TYPE_PADD}).SetOutput(PATTERN_PADD);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define PaddUpdateFusionPass pattern");
  return patterns;
}

Status PaddUpdateFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into PaddUpdateFusionPass");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get PadD node"),
                    return PARAM_INVALID);

  ge::OpDescPtr fusedOpDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get op desc"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(fusedOpDesc->GetInputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Exit PaddUpdateFusionPass due to input format is NC1HWC0"),
                    return NOT_CHANGED);

  std::vector<std::vector<int64_t>> paddings;
  ge::AttrUtils::GetListListInt(fusedOpDesc, PADDINGS, paddings);
  FUSION_PASS_CHECK(paddings.size() < 1 || paddings[0].size() < 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get paddings value from PadD node"), return NOT_CHANGED);

  ge::GeShape constShape = ge::GeShape({paddings.size(), paddings[0].size()});
  auto constTensorDesc = ge::GeTensorDesc(constShape, ge::FORMAT_ND, ge::DT_INT64);
  std::vector<int64_t> constValue;
  for (size_t i = 0; i < paddings.size(); i++) {
    constValue.insert(constValue.end(), paddings[i].begin(), paddings[i].end());
  }
  ge::GeTensorPtr constTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(constTensor = std::make_shared<ge::GeTensor>(constTensorDesc), return INTERNAL_ERROR);
  constTensor->SetData(reinterpret_cast<uint8_t*>(constValue.data()), constValue.size() * sizeof(int64_t));
  ge::OpDescPtr constOpDesc = ge::OpDescUtils::CreateConstOp(constTensor);
  auto constNode = graph.AddNode(constOpDesc);
  newNodes.push_back(constNode);

  std::string padNodeName = fusedNode->GetName() + "/Pad";
  ge::OpDescPtr padOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(padOpDesc = std::make_shared<ge::OpDesc>(padNodeName, "Pad"), return INTERNAL_ERROR);
  FUSION_PASS_CHECK(padOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Pad op desc"),
                    return FAILED);

  FUSION_PASS_CHECK(padOpDesc->AddInputDesc(0, fusedOpDesc->GetInputDesc(0)) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x desc for Pad"),
                    return FAILED);
  FUSION_PASS_CHECK(padOpDesc->AddInputDesc(1, constTensorDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add paddings desc for Pad"),
                    return FAILED);
  FUSION_PASS_CHECK(padOpDesc->AddOutputDesc(fusedOpDesc->GetOutputDesc(0)) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add y desc for Pad"),
                    return FAILED);
  ge::NodePtr padNode = graph.AddNode(padOpDesc);
  newNodes.push_back(padNode);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            padNode->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from input node to Pad"),
                    return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(constNode->GetOutDataAnchor(0), padNode->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from Const to Pad"), return FAILED);
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  if (fusedNode->GetInControlAnchor() != nullptr) {
    if (!fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().empty()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The PeerOutControlAnchors of fused node[%s] input control anchor is not empty",
              fusedNode->GetName().c_str());
      for (auto outAnchor : fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchor, fusedNode->GetInControlAnchor()) != ge::GRAPH_SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add in control edge to Pad"),
                          return FAILED);
      }
    }
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto inAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(padNode->GetOutDataAnchor(0), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from Pad to output node"),
        return FAILED);
  }
  if (fusedNode->GetOutControlAnchor() != nullptr) {
    if (!fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The PeerInControlAnchors of fused node[%s] output control anchor is not empty",
              fusedNode->GetName().c_str());
      for (InControlAnchorPtr inCtrlAnchorPtr : fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(padNode->GetOutControlAnchor(), inCtrlAnchorPtr),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add output control edge for Pad"),
            return FAILED);
      }
    }
    fusedNode->GetOutControlAnchor()->UnlinkAll();
  }

  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove Node:%s", fusedNode->GetName().c_str()),
      return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to PaddUpdateFusionPass");
  return SUCCESS;
}

REGISTER_PASS("PaddUpdateFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, PaddUpdateFusionPass);
}  // namespace fe
