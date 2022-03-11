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
 * \file concat_v2_fusion_pass.cpp
 * \brief ConcatV2 -> ConcatV2D && split big ConcatV2D into small ones
 */
#include "concat_v2_fusion_pass.h"

#include <iostream>
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
#include "concat_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "ConcatV2";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcatV2";

vector<FusionPattern*> ConcatExt2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatExt2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
// including newly added nodes and fused but not deleted nodes
Status ConcatExt2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  std::string fusion_op_type = "ConcatV2D";
  std::vector<PassAttrInfo> concatv2_attr_info;
  ge::NodePtr fused_node = nullptr;
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new a pattern object failed"),
                    return PARAM_INVALID);
  int32_t inputsize = fused_node1->GetAllInDataAnchors().size();
  PassAttrInfo axis = {inputsize - 1, "concat_dim", "SetInt"};
  concatv2_attr_info.push_back(axis);

  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusion_op_type, concatv2_attr_info, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE, "Concatv2 has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  ClearOpInferDepends(fused_node1);
  OP_LOGI(fused_node->GetName(), "Concatv2-->ConcatV2D fusion SUCCESSS!!!!!");

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(
      fused_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fused_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);

  FUSION_PASS_CHECK(
      RemoveInvalidEdge(fused_node, fused_desc, FUSED_OP_TYPE) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "delete invalid edges failed."),
      return FAILED);

  // check whether do change or not
  int64_t max_inputs = GetMaxInputsNum(fused_node);
  FUSION_PASS_CHECK(
      !CheckNeedChanged(fused_desc, max_inputs),
      OP_LOGD(FUSED_OP_TYPE, "The amount of input of ConcatV2D node is less than %lld.", max_inputs);
      new_nodes.emplace_back(fused_node), return SUCCESS);

  // split concatv2d node
  FUSION_PASS_CHECK(
      SplitConcatNode(graph, new_nodes, fused_node, max_inputs, FUSED_OP_TYPE) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "split base concatv2d failed."),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("ZConcatExt2FusionPass", BUILT_IN_GRAPH_PASS, ConcatExt2FusionPass);
}  // namespace fe
