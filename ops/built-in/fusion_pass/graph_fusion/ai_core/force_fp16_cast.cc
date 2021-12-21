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
 * \file force_fp16_cast.cc
 * \brief force_fp16_cast fusion pass
 */
#include "force_fp16_cast.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
/*
          node1(aicore support or nor support)               node1(if aicore support set attr _keep_dtype=1)
            |                                                  |
            |                                                  |
            |                                                  |
          cast(fp32->int32 or int32->fp32,            ------>cast(set attr _keep_dtype=1)
            |  aicore support)                                 |
            |                                                  |
            |                                                  |
            |                                                  |
          node2(aicore support)                              node2(set attr _keep_dtype=1)
*/
static const string PATTERN_CAST = "Cast";
vector<FusionPattern*> ForceFp16CastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ForceFp16CastFusionPassPattern");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CAST, {"Cast"}).SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "patterns ForceFp16CastFusionPass is end");
  return patterns;
}

Status ForceFp16CastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ForceFp16CastFusionPass is running.");
  ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(cast_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "cast_node node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr cast_desc = cast_node->GetOpDesc();
  if (cast_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to get op desc cast_desc");
    return FAILED;
  }
  const std::map<ge::DataType, ge::DataType> in_out_dtype = {{DT_FLOAT, DT_INT32}, {DT_INT32, DT_FLOAT}};
  auto find_in = in_out_dtype.find(cast_desc->GetInputDesc(0).GetDataType());
  auto find_out = in_out_dtype.find(cast_desc->GetOutputDesc(0).GetDataType());
  if (find_in == in_out_dtype.end() || find_out == in_out_dtype.end()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Cast input dtype is not int32 , fp32, or output is not fp32, int32");
    return NOT_CHANGED;
  }

  // get pre node of cast
  ge::InDataAnchorPtr cast_anchor_tr0 = cast_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(
      cast_anchor_tr0 == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast_anchor_tr0's OpDesc is null, fusion failed."),
      return PARAM_INVALID);
  ge::OutDataAnchorPtr pre_anchor_ptr0 = cast_anchor_tr0->GetPeerOutAnchor();
  ge::NodePtr pre_node = pre_anchor_ptr0->GetOwnerNode();
  ge::OpDescPtr pre_desc = pre_node->GetOpDesc();
  bool support_pre = false;
  bool support_cast = false;
  bool support_next = false;
  int64_t keep_dtype = 1;
  support_pre = CheckOpSupported(pre_desc);
  support_cast = CheckOpSupported(cast_desc);

  // get next node of cast
  ge::NodePtr next_node = nullptr;
  if (cast_node->GetAllOutDataAnchors().empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Cast output is null");
    return PARAM_INVALID;
  }
  if (cast_node->GetAllOutDataAnchors().size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Cast output is not one");
    return NOT_CHANGED;
  }
  for (OutDataAnchorPtr out_data_anchor : cast_node->GetAllOutDataAnchors()) {
    if (out_data_anchor == nullptr) {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "out_data_anchor Node[%s] has a "
              "nullptr Out Data anchor.",
              cast_node->GetName().c_str());
      return NOT_CHANGED;
    }
    if (out_data_anchor->GetPeerInDataAnchors().empty()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "GetPeerInDataAnchors Node[%s] has a nullptr "
              "Out Data anchor.",
              cast_node->GetName().c_str());
      return NOT_CHANGED;
    }
    for (InDataAnchorPtr in_data_anchorPtr : out_data_anchor->GetPeerInDataAnchors()) {
      if (in_data_anchorPtr == nullptr) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr in Data anchor.", cast_node->GetName().c_str());
        return NOT_CHANGED;
      }
      next_node = in_data_anchorPtr->GetOwnerNode();
      ge::OpDescPtr next_desc = next_node->GetOpDesc();
      support_next = CheckOpSupported(next_desc);
    }
  }
  if (support_cast && support_next && support_pre) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "support_next support_cast support_pre is all support");
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(cast_node->GetOpDesc(), "_keep_dtype", keep_dtype),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", "mode",
                                                     cast_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(next_node->GetOpDesc(), "_keep_dtype", keep_dtype),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", "mode",
                                                     next_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(pre_node->GetOpDesc(), "_keep_dtype", keep_dtype),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", "mode",
                                                     pre_node->GetName().c_str()),
                      return FAILED);
  } else if (support_cast && support_next && (!support_pre)) {
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(cast_node->GetOpDesc(), "_keep_dtype", keep_dtype),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", "mode",
                                                     cast_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(next_node->GetOpDesc(), "_keep_dtype", keep_dtype),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", "mode",
                                                     next_node->GetName().c_str()),
                      return FAILED);
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "else not changed");
    return NOT_CHANGED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ForceFp16CastFusionPass run success.");
  return SUCCESS;
}

REGISTER_PASS("ForceFp16CastFusionPass", BUILT_IN_GRAPH_PASS, ForceFp16CastFusionPass);
}  // namespace fe
