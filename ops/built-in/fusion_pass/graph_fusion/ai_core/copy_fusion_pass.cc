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
 * \file copy_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop \brief LayerNormBetaGammaBackprop)
 */
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
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

#include "copy_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_COPY = "Copy";

vector<FusionPattern*> CopyFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("CopyFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_COPY, {"Copy"}).SetOutput(PATTERN_COPY);

  patterns.push_back(pattern);

  return patterns;
}

Status CopyFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // Get copy node and copy node description.
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into COPYPass");
  ge::NodePtr copyNode = GetNodeFromMapping(PATTERN_COPY, mapping);
  FUSION_PASS_CHECK(copyNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "copyNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr copyDesc = copyNode->GetOpDesc();
  FUSION_PASS_CHECK(copyDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "copyNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // Get Input Node
  ge::InDataAnchorPtr oriInAnchorPtr0 = copyNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputNode = oriBottomPeerAnchorPtr0->GetOwnerNode();

  // Get oriTopPeerAnchorPtrs
  vector<ge::InDataAnchorPtr> oriTopPeerAnchorPtrs;
  for (auto outAnchor : copyNode->GetAllOutDataAnchors()) {
    for (auto dstInAnchor : outAnchor->GetPeerInDataAnchors()) {
      oriTopPeerAnchorPtrs.push_back(dstInAnchor);
    }
  }

  for (auto inAnchor : copyNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  if (copyNode->GetInControlAnchor() != nullptr) {
    copyNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : copyNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  if (copyNode->GetOutControlAnchor() != nullptr) {
    copyNode->GetOutControlAnchor()->UnlinkAll();
  }

  for (uint64_t i = 0; i < oriTopPeerAnchorPtrs.size(); i++) {
    ge::NodePtr outputNode = oriTopPeerAnchorPtrs.at(i)->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, oriTopPeerAnchorPtrs.at(i)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              inputNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(copyNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove copy node failed"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "COPYPass success!!!!");

  return SUCCESS;
}
REGISTER_PASS("COPYPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, CopyFusionPass);
}  // namespace fe
