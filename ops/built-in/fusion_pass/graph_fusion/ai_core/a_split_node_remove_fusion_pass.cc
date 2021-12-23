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
 * \file a_split_node_remove_fusion_pass.cpp
 * \brief
 */
#include "a_split_node_remove_fusion_pass.h"

#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "Split";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSplit";
/*
1:
Conv2D + Split(num_split = 1) ---> Conv2D
*/
vector<FusionPattern*> SplitNodeRemoveFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SplitNodeRemoveFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status SplitNodeRemoveFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SplitNodeRemoveFusionPass begin.");
  // get node
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed"),
                    return PARAM_INVALID);

  // get desc
  ge::OpDescPtr fuse_desc1 = fused_node1->GetOpDesc();
  FUSION_PASS_CHECK(
      fuse_desc1 == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fused_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);

  int64_t num_split;
  ge::AttrUtils::GetInt(fuse_desc1, "num_split", num_split);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "The value of num_split is %d.", num_split);

  if (num_split != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The value of num_split is not equal 1, graph not changed.");
    return NOT_CHANGED;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start delete edges of split, and connect the beforeNode to afterNode.");
  for (auto inDataAnchor : fused_node1->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node1->GetOutDataAnchor(0), inDataAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node1->GetInDataAnchor(1)->GetPeerOutAnchor(), inDataAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node1->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                          fused_node1->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input_data[0] edge failed."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start delete split node.");
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node1),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node1->GetName().c_str()),
      return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove split node success when num_split = 1.");
  return SUCCESS;
}

REGISTER_PASS("ASplitNodeRemoveFusionPass", BUILT_IN_GRAPH_PASS, SplitNodeRemoveFusionPass);
}  // namespace fe
