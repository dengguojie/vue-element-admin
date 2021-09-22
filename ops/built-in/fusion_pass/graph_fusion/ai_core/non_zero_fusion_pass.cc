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
 * \file non_zero_fusion_pass.cpp
 * \brief NonZero fusion pass
 */
#include "non_zero_fusion_pass.h"

#include <memory>
#include <string>

#include "anchor_util.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_NONZERO = "NonZero";
static const string NONZERO = "NonZero";
static const string PATTERN_CAST = "Cast";
static const string CAST = "Cast";
vector<FusionPattern*> NonZeroFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define NonZeroFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("NonZeroFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                             "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_NONZERO, {NONZERO})
      .AddOpDesc(PATTERN_CAST, {CAST})
      .SetInputs(PATTERN_CAST, {PATTERN_NONZERO})
      .SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define NonZeroFusionPass pattern end");
  return patterns;
}

Status NonZeroFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define NonZeroFusionPass fusion begin");
  ge::NodePtr nonZeroNode = GetNodeFromMapping(PATTERN_NONZERO, mapping);
  FUSION_PASS_CHECK(nonZeroNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NonZero node is null"),
                    return PARAM_INVALID);

  ge::NodePtr castNode = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(castNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Cast node is null"),
                    return PARAM_INVALID);

  if (castNode->GetOpDesc()->GetOutputDesc(0).GetDataType() != ge::DT_INT32) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "No need change graph.");
      return FAILED;
  }

  nonZeroNode->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);

  // add edge nonzero node link cast outputs
  ge::OutDataAnchorPtr castNodePtr = castNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(castNodePtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "castNodePtr is null."),
                    return FAILED);
  auto inAnchors = castNodePtr->GetPeerInDataAnchors();
  for (auto inAnchor : inAnchors) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(nonZeroNode->GetOutDataAnchor(0), inAnchor) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Adding edge from fusion node: %s's index[0] to fused node: %s's indexes is failed.",
              nonZeroNode->GetName().c_str(), castNode->GetName().c_str()),
      return FAILED);
  }

  // remove cast node inEdge
  for (auto inAnchor : castNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define NonZeroFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("NonZeroFusionPass", BUILT_IN_GRAPH_PASS, NonZeroFusionPass);
}  // namespace fe
