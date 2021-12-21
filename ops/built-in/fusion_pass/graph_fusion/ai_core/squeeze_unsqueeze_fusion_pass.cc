/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file squeeze_unsqueeze_fusion_pass.cc
 * \brief Clear redundant squeeze + unsqueeze nodes.
 */
#include "squeeze_unsqueeze_fusion_pass.h"

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

using namespace ge;
namespace fe {
static const char PATTERN_SQUEEZE[] = "Squeeze";
static const char PATTERN_UNSQUEEZE[] = "Unsqueeze";

static const char OPTYPE_SQUEEZE[] = "Squeeze";
static const char OPTYPE_UNSQUEEZE[] = "Unsqueeze";

vector<FusionPattern*> SqueezeUnsqueezeFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern begin");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SqueezeUnsqueezeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SQUEEZE, {OPTYPE_SQUEEZE})
      .AddOpDesc(PATTERN_UNSQUEEZE, {OPTYPE_UNSQUEEZE})
      .SetInputs(PATTERN_UNSQUEEZE, {PATTERN_SQUEEZE})
      .SetOutput(PATTERN_UNSQUEEZE);

  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern end");
  return patterns;
}

Status SqueezeUnsqueezeFusionPass::CheckParams(const ge::NodePtr& squeezeNode, const ge::NodePtr& unsqueezeNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckParams begin");

  FUSION_PASS_CHECK(squeezeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Squeeze node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr squeezeDesc = squeezeNode->GetOpDesc();
  FUSION_PASS_CHECK(
      squeezeDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OpDesc of squeeze node is null, fusion failed."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(squeezeNode->GetAllOutAnchors().size() != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The Squeeze node should only have 1 output anchor."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      squeezeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1,
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The Squeeze node should only have 1 output anchor that link to other nodes."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      squeezeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != "Unsqueeze",
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The Squeeze node should only have 1 output anchor that link to Unsqueeze."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(unsqueezeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Unsqueeze node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr unsqueezeDesc = unsqueezeNode->GetOpDesc();
  FUSION_PASS_CHECK(
      unsqueezeDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OpDesc of unsqueeze node is null, fusion failed."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(unsqueezeNode->GetAllInDataAnchors().size() != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The unsqueeze node should only have 1 input anchor."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(unsqueezeNode->GetAllOutDataAnchors().size() != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The unsqueeze node should only have 1 output anchor."),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Squeeze node name: %s. Unsqueeze node name: %s", squeezeNode->GetName().c_str(),
          unsqueezeNode->GetName().c_str());

  FUSION_PASS_CHECK(!squeezeNode->GetInControlNodes().empty() || !squeezeNode->GetOutControlNodes().empty(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Squeeze node has control edge."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!unsqueezeNode->GetInControlNodes().empty() || !unsqueezeNode->GetOutControlNodes().empty(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Unsqueeze node has control edge."), return NOT_CHANGED);

  ge::GeShape squeezeOriginShape = squeezeDesc->GetInputDesc(0).GetOriginShape();
  ge::GeShape squeezeShape = squeezeDesc->GetInputDesc(0).GetShape();
  ge::GeShape unsqueezeOriginShape = unsqueezeDesc->GetOutputDesc(0).GetOriginShape();
  ge::GeShape unsqueezeShape = unsqueezeDesc->GetOutputDesc(0).GetShape();
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "squeezeOriginShape: %s. squeezeShape: %s. unsqueezeOriginShape: %s. unsqueezeShape: %s.",
          squeezeOriginShape.ToString().c_str(), squeezeShape.ToString().c_str(),
          unsqueezeOriginShape.ToString().c_str(), unsqueezeShape.ToString().c_str());
  FUSION_PASS_CHECK(squeezeOriginShape.GetDimNum() != 4 || squeezeShape.GetDimNum() != 4 ||
                        unsqueezeOriginShape.GetDimNum() != 4 || unsqueezeShape.GetDimNum() != 4,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The OriginShape/Shape size of squeeze/unsqueeze node should be 4."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(squeezeOriginShape.GetDim(0) != unsqueezeOriginShape.GetDim(0) ||
                        squeezeOriginShape.GetDim(1) != unsqueezeOriginShape.GetDim(1),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The previous dim should be the same except for the last 2 dim."),
                    return NOT_CHANGED);

  /**
   * Support cases are such as:
   *
   * | no. |   squeeze shape  |  unsqueeze shape |
   * |-----|------------------|------------------|
   * |  1  | (8, 160, 63,  1) | (8, 160,  1, 63) |
   * |  2  | (8, 160, 63,  1) | (8, 160, 63,  1) |
   * |  3  | (8, 160,  1, 63) | (8, 160,  1, 63) |
   * |  4  | (8, 160, 63,  1) | (8, 160, 63,  1) |
   * |  5  | (8, 160,  1,  1) | (8, 160,  1,  1) |
   */
  FUSION_PASS_CHECK(squeezeOriginShape.GetDim(2) != 1 && squeezeOriginShape.GetDim(3) != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "One of the last two dimensions of squeeze should be 1."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(unsqueezeOriginShape.GetDim(2) != 1 && unsqueezeOriginShape.GetDim(3) != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "One of the last two dimensions of unsqueeze should be 1."),
                    return NOT_CHANGED);

  int64_t squeezeLastDim = squeezeOriginShape.GetDim(2);
  if (squeezeLastDim == 1) {
    squeezeLastDim = squeezeOriginShape.GetDim(3);
  }
  int64_t unsqueezeLastDim = unsqueezeOriginShape.GetDim(2);
  if (unsqueezeLastDim == 1) {
    unsqueezeLastDim = unsqueezeOriginShape.GetDim(3);
  }
  FUSION_PASS_CHECK(squeezeLastDim != unsqueezeLastDim,
                    OP_LOGD(FUSED_OP_TYPE.c_str(),
                            "Data ranking is inconsistent for the last dim of squeeze and unqueeze are different."),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckParams end");
  return SUCCESS;
}

Status SqueezeUnsqueezeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion begin");

  ge::NodePtr squeezeNode = GetNodeFromMapping(PATTERN_SQUEEZE, mapping);
  ge::NodePtr unsqueezeNode = GetNodeFromMapping(PATTERN_UNSQUEEZE, mapping);
  Status checkRet = CheckParams(squeezeNode, unsqueezeNode);
  FUSION_PASS_CHECK(SUCCESS != checkRet, OP_LOGI(FUSED_OP_TYPE.c_str(), "Unsupport parameters. Fusion end."),
                    return checkRet);

  if (unsqueezeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The output edge of unsqueeze is [%d].",
            unsqueezeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : unsqueezeNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(squeezeNode->GetInDataAnchor(0)->GetPeerOutAnchor(), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input to fusion node:%s's output failed.",
                  squeezeNode->GetName().c_str(), unsqueezeNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input to fusion node:%s's output success.",
              squeezeNode->GetName().c_str(), unsqueezeNode->GetName().c_str());
    }
  }

  FUSION_PASS_CHECK(graph.RemoveNode(squeezeNode) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove squeeze node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(unsqueezeNode) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove unsqueeze node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion end");
  return SUCCESS;
}

REGISTER_PASS("SqueezeUnsqueezeFusionPass", BUILT_IN_GRAPH_PASS, SqueezeUnsqueezeFusionPass);
}  // namespace fe
