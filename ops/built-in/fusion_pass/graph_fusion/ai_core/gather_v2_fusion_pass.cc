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
 * \file gather_v2_fusion_pass.cpp
 * \brief split fusion pass(gather_v2 --> gather_v2_d)
 */
#include "gather_v2_fusion_pass.h"

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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "register/op_registry.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "GatherV2";
static const char CAST[] = "Cast";

static const std::string PATTERN_FUSEDNODE = "FusedNodeGatherV2";

vector<FusionPattern*> ConstToAttrGatherV2Pass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrGatherV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrGatherV2Pass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "GatherV2D";
  std::vector<PassAttrInfo> gather_v2AttrInfo;
  PassAttrInfo axis = {2, "axis", "SetInt"};
  gather_v2AttrInfo.push_back(axis);

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                   fusedNode->GetName().c_str()),
                    return PARAM_INVALID);

  if ((fusedNode->GetInDataAnchor(1) != nullptr) && (fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor() != nullptr)) {
    if ((fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() != nullptr) &&
        (fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == CAST)) {
      ge::NodePtr cast_node = fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
      ge::OpDescPtr castDesc = cast_node->GetOpDesc();
      FUSION_PASS_CHECK(
          castDesc == nullptr,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                         cast_node->GetName().c_str()),
          return PARAM_INVALID);
      ge::OutDataAnchorPtr inAnchorOfCast = cast_node->GetInDataAnchor(0)->GetPeerOutAnchor();
      ge::OutDataAnchorPtr outAnchorOfCast = cast_node->GetOutDataAnchor(0);
      if ((castDesc->GetInputDesc(0).GetDataType() == ge::DT_INT64) &&
          (castDesc->GetOutputDesc(0).GetDataType() == ge::DT_INT32)) {
        // remove cast output edge
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfCast, fusedNode->GetInDataAnchor(1)) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove cast output edge failed."),
                          return FAILED);
        // add edge
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                  fusedNode->GetInDataAnchor(1)) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge failed."), return FAILED);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "start to remove cast node.");
        if (cast_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty()) {
          // remove cast input edge
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(inAnchorOfCast, cast_node->GetInDataAnchor(0)) != SUCCESS,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove cast input edge failed."),
                            return FAILED);
          FUSION_PASS_CHECK(graph.RemoveNode(cast_node) != SUCCESS,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove cast node failed."),
                            return FAILED);
          OP_LOGI(FUSED_OP_TYPE.c_str(), "remove cast node success.");
        }
        fusedDesc->MutableInputDesc(1)->SetDataType(ge::DT_INT64);
        fusedDesc->MutableInputDesc(1)->SetOriginDataType(ge::DT_INT64);
      }
    }
  }

  // check op support for dynamic gather_v2
  FUSION_PASS_CHECK(CheckOpSupported(fusedDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Gatherv2 Supported."),
                    return NOT_CHANGED);

  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, gather_v2AttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return NOT_CHANGED);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Gatherv2 Not Supported."),
                    return NOT_CHANGED);

  // PatternFusionUtil patternFusionUtil;
  if (domi::OpRegistry::Instance()->GetImplyType(fusedDesc->GetType()) == domi::ImplyType::CCE) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fusion Op %s is CCE, no need to fusion.", fusedNode->GetName().c_str());
    return NOT_CHANGED;
  }
  ge::NodePtr fusion_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, gather_v2AttrInfo, fusion_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "GatherV2 has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusion_node);
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrGatherV2Fusion", BUILT_IN_GRAPH_PASS, ConstToAttrGatherV2Pass);
}  // namespace fe
