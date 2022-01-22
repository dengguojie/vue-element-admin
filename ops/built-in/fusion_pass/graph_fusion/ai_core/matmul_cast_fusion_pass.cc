/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file matmul_cast_fusion_pass.cpp
 * \brief matmul cast fusion (MatMul--Cast)
 */
#include "matmul_cast_fusion_pass.h"
#include <memory>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "error_util.h"

namespace fe {
static const string PATTERN_MATMUL = "matmul";
static const string PATTERN_CAST = "cast";

/*
    fusion pattern
            node
                \
                 \
                Matmul---Cast---
                /
               /
            node
*/
vector<FusionPattern*> MatmulCastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatMulCastFusion");
  if (pattern == nullptr) {
    OP_LOGE(FUSED_OP_TYPE, "pattern is nullptr, Create pattern not success!");
    return patterns;
  }
  pattern->AddOpDesc(PATTERN_MATMUL, {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"})
      .AddOpDesc(PATTERN_CAST, {"Cast"})
      .SetInputs(PATTERN_CAST, {PATTERN_MATMUL})
      .SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);
  return patterns;
}

Status MatmulCastFusionPass::IsMatch(const ge::NodePtr &matmul_node, const ge::NodePtr &cast_node) const {
  if (matmul_node->GetOutDataNodes().size() != 1) {
    OP_LOGD(FUSED_OP_TYPE, "MatMul outputs num shoubld be 1");
    return FAILED;
  }
  ge::DataType matmul_output_dtype = matmul_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType cast_output_dtype = cast_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (matmul_output_dtype != ge::DT_FLOAT16 || cast_output_dtype != ge::DT_FLOAT) {
    OP_LOGD(FUSED_OP_TYPE, "MatMul output dtype is %u, Cast output dtype is %u", matmul_output_dtype,
            cast_output_dtype);
    return FAILED;
  }
  return SUCCESS;
}

Status MatmulCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE, "Enter MatmulCastFusionPass.");
  ge::NodePtr matmul_node = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGE(FUSED_OP_TYPE, "MatMul node is null"), return FAILED);
  FUSION_PASS_CHECK(cast_node == nullptr, OP_LOGE(FUSED_OP_TYPE, "Cast node is null"), return FAILED);

  if (IsMatch(matmul_node, cast_node) != SUCCESS) {
    OP_LOGD(FUSED_OP_TYPE, "Node[%s] and node[%s] don't match Matmul + Cast fusion pattern.",
            matmul_node->GetName().c_str(), cast_node->GetName().c_str());
    return NOT_CHANGED;
  }

  DoFusion(matmul_node);
  FUSION_PASS_CHECK(!CheckOpSupported(matmul_node->GetOpDesc()),
                    OP_LOGI(FUSED_OP_TYPE, "MatMul[%s] is not supported by FE, fusion abort.",
                            matmul_node->GetOpDesc()->GetName().c_str()),
                    return NOT_CHANGED);

  // link matmul output with cast output and remove cast node
  if (LinkOutputEdgeWithoutControl(matmul_node, cast_node) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE, "link output edge Failed.");
    return FAILED;
  }
  if (graph.RemoveNode(cast_node) == ge::GRAPH_FAILED) {
    OP_LOGE(FUSED_OP_TYPE, "cast node remove failed");
    return FAILED;
  }

  OP_LOGD(FUSED_OP_TYPE, "Node[%s] do MatMul + Cast fusion success!", matmul_node->GetName().c_str());
  return SUCCESS;
}

Status MatmulCastFusionPass::DoFusion(const ge::NodePtr &matmul_node) const {
  auto matmul_output_desc = matmul_node->GetOpDesc()->MutableOutputDesc(0);
  matmul_output_desc->SetDataType(ge::DT_FLOAT);
  matmul_output_desc->SetOriginDataType(ge::DT_FLOAT);

  return SUCCESS;
}

Status MatmulCastFusionPass::LinkOutputEdgeWithoutControl(const ge::NodePtr& matmul_node,
                                                          const ge::NodePtr& cast_node) const {
  // Remove cast node all input edge
  if (PatternFusionUtil::RemoveInputEdge(cast_node) == FAILED) {
    OP_LOGE(FUSED_OP_TYPE, "Remove cast input edge Failed.");
    return FAILED;
  }
  auto matmul_out_anchor = matmul_node->GetOutDataAnchor(0);
  if (matmul_out_anchor == nullptr) {
    OP_LOGE(FUSED_OP_TYPE, "Parameter[matmul_out_anchor] must not be null.");
    return FAILED;
  }
  // Remove cast->output anchor and add matmul->output anchor
  for (ge::OutDataAnchorPtr &anchor : cast_node->GetAllOutDataAnchors()) {
    if (anchor != nullptr) {
      for (ge::InDataAnchorPtr &dst_anchor : anchor->GetPeerInDataAnchors()) {
        if (ge::GraphUtils::RemoveEdge(anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
            ge::GraphUtils::AddEdge(matmul_out_anchor, dst_anchor) != ge::GRAPH_SUCCESS) {
          OP_LOGE(FUSED_OP_TYPE, "Replace out data anchor Failed.");
          return FAILED;
        }
      }
    }
  }
  // Remove cast->output control anchor and add matmul->output control anchor
  auto cast_out_control_anchor = cast_node->GetOutControlAnchor();
  if (cast_out_control_anchor != nullptr) {
    for (ge::InControlAnchorPtr &dst_anchor : cast_out_control_anchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(cast_out_control_anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(matmul_node->GetOutControlAnchor(), dst_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE, "Replace out control anchor Failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

REGISTER_PASS("MatmulCastFusionPass", BUILT_IN_GRAPH_PASS, MatmulCastFusionPass);
REGISTER_PASS("MatmulCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, MatmulCastFusionPass);
}  // namespace fe
