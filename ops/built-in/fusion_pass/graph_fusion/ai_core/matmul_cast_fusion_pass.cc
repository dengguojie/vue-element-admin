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
static const char kFusionName[] = "MatmulCastFusionPass";
static const string PATTERN_MATMUL = "matmul";
static const string PATTERN_CAST = "cast";
static const int32_t kInputNum = 2;

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
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kFusionName, "Create pattern fail!"), return patterns);
  pattern->AddOpDesc(PATTERN_MATMUL, {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"})
      .AddOpDesc(PATTERN_CAST, {"Cast"})
      .SetInputs(PATTERN_CAST, {PATTERN_MATMUL})
      .SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);
  return patterns;
}

Status MatmulCastFusionPass::IsMatch(const ge::NodePtr &matmul_node, const ge::NodePtr &cast_node) const {
  if (matmul_node->GetOutDataNodes().size() != 1) {
    OP_LOGD(kFusionName, "MatMul outputs num shoubld be 1");
    return FAILED;
  }
  ge::DataType matmul_output_dtype = matmul_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType cast_output_dtype = cast_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (matmul_output_dtype != ge::DT_FLOAT16 || cast_output_dtype != ge::DT_FLOAT) {
    OP_LOGD(kFusionName, "MatMul output dtype is %u, Cast output dtype is %u", matmul_output_dtype,
            cast_output_dtype);
    return FAILED;
  }

  // check dynamic shape
  auto matmul_desc = matmul_node->GetOpDesc();
  if (matmul_desc->MutableInputDesc(0)->MutableShape().IsUnknownShape() ||
      matmul_desc->MutableInputDesc(1)->MutableShape().IsUnknownShape()) {
    OP_LOGD(kFusionName, "Do not support dynamic shape.");
    return FAILED;
  }
  if (matmul_node->GetInDataNodes().size() > kInputNum) {
    if (matmul_desc->MutableInputDesc(kInputNum)->MutableShape().IsUnknownShape()) {
      OP_LOGD(kFusionName, "Do not support dynamic shape.");
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatmulCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(kFusionName, "Enter MatmulCastFusionPass.");
  ge::NodePtr matmul_node = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGW(matmul_node, "MatMul node is null"), return NOT_CHANGED);
  FUSION_PASS_CHECK(cast_node == nullptr, OP_LOGW(cast_node, "Cast node is null"), return NOT_CHANGED);

  if (IsMatch(matmul_node, cast_node) != SUCCESS) {
    OP_LOGD(kFusionName, "Node[%s] and node[%s] don't match Matmul + Cast fusion pattern.",
            matmul_node->GetName().c_str(), cast_node->GetName().c_str());
    return NOT_CHANGED;
  }

  ge::DataType matmul_output_dtype;
  ge::DataType matmul_output_ori_dtype;
  DoFusion(matmul_node, matmul_output_dtype, matmul_output_ori_dtype);
  FUSION_PASS_CHECK(!CheckOpSupported(matmul_node->GetOpDesc()),
                    OP_LOGW(matmul_node, "MatMul[%s] is not supported by FE, fusion abort.",
                            matmul_node->GetOpDesc()->GetName().c_str());
                    RestoreDtype(matmul_node, matmul_output_dtype, matmul_output_ori_dtype), return NOT_CHANGED);

  // link matmul output with cast output and remove cast node
  FUSION_PASS_CHECK(LinkOutputEdgeWithoutControl(matmul_node, cast_node) == FAILED,
                    OP_LOGE(matmul_node, "link output edge Failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(cast_node) == ge::GRAPH_FAILED, OP_LOGE(cast_node, "cast node remove failed"),
                    return FAILED);

  OP_LOGI(matmul_node, "Node[%s] do MatMul + Cast fusion success!", matmul_node->GetName().c_str());
  return SUCCESS;
}

Status MatmulCastFusionPass::DoFusion(const ge::NodePtr &matmul_node,  ge::DataType &matmul_output_dtype,
                                      ge::DataType &matmul_output_ori_dtype) const {
  auto matmul_output_desc = matmul_node->GetOpDesc()->MutableOutputDesc(0);
  matmul_output_dtype = matmul_output_desc->GetDataType();
  matmul_output_ori_dtype = matmul_output_desc->GetOriginDataType();
  matmul_output_desc->SetDataType(ge::DT_FLOAT);
  matmul_output_desc->SetOriginDataType(ge::DT_FLOAT);

  return SUCCESS;
}

void MatmulCastFusionPass::RestoreDtype(ge::NodePtr &matmul_node, const ge::DataType &matmul_output_dtype,
                                        const ge::DataType &matmul_output_ori_dtype) const {
  auto matmul_output_desc = matmul_node->GetOpDesc()->MutableOutputDesc(0);
  matmul_output_desc->SetDataType(matmul_output_dtype);
  matmul_output_desc->SetOriginDataType(matmul_output_ori_dtype);
}

Status MatmulCastFusionPass::LinkOutputEdgeWithoutControl(const ge::NodePtr& matmul_node,
                                                          const ge::NodePtr& cast_node) const {
  // Remove cast node all input edge
  FUSION_PASS_CHECK(PatternFusionUtil::RemoveInputEdge(cast_node) == FAILED,
                    OP_LOGE(cast_node, "Remove cast input edge Failed."),
                    return FAILED);
  auto matmul_out_anchor = matmul_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(matmul_out_anchor == nullptr,
                    OP_LOGE(matmul_node, "Parameter[matmul_out_anchor] must not be null."),
                    return FAILED);
  // Remove cast->output anchor and add matmul->output anchor
  for (ge::OutDataAnchorPtr &anchor : cast_node->GetAllOutDataAnchors()) {
    FUSION_PASS_CHECK(anchor == nullptr, OP_LOGE(cast_node, "Parameter[cast_out_anchor] must not be null."),
                      return FAILED);
    for (ge::InDataAnchorPtr &dst_anchor : anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(matmul_out_anchor, dst_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(cast_node, "Replace out data anchor Failed.");
        return FAILED;
      }
    }
  }
  // Remove cast->output control anchor and add matmul->output control anchor
  auto cast_out_control_anchor = cast_node->GetOutControlAnchor();
  if (cast_out_control_anchor != nullptr) {
    for (ge::InControlAnchorPtr &dst_anchor : cast_out_control_anchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(cast_out_control_anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(matmul_node->GetOutControlAnchor(), dst_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(cast_node, "Replace out control anchor Failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

REGISTER_PASS("MatmulCastFusionPass", BUILT_IN_GRAPH_PASS, MatmulCastFusionPass);
REGISTER_PASS("MatmulCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, MatmulCastFusionPass);
}  // namespace fe
