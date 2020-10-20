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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "register/op_registry.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "GatherV2";

static const std::string PATTERN_FUSEDNODE = "FusedNodeGatherV2";

vector<FusionPattern*> ConstToAttrGatherV2Pass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrGatherV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
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
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(
      fusedDesc == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);

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
