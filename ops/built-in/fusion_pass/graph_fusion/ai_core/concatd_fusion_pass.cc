/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
 * \file concatd_fusion_pass.cpp
 * \brief split big ConcatD into small ones
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concatd_fusion_pass.h"
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
#include "concat_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "ConcatD";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";
vector<FusionPattern*> ConcatDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                    "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &new_nodes: Store fusion nodes,
// including newly added nodes and fused but not deleted nodes
Status ConcatDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                    "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // check whether do change or not
  int64_t max_inputs = GetMaxInputsNum(fused_node);
  FUSION_PASS_CHECK(
      !CheckNeedChanged(fused_desc, max_inputs),
      OP_LOGD(FUSED_OP_TYPE, "The amount of input of ConcatD node is less than %lld.", max_inputs),
      return NOT_CHANGED);

  // split concatd node
  FUSION_PASS_CHECK(
      SplitConcatNode(graph, new_nodes, fused_node, max_inputs, FUSED_OP_TYPE) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "split base concatd failed."),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("ZConcatDFusionPass", BUILT_IN_GRAPH_PASS, ConcatDFusionPass);
}  // namespace fe
