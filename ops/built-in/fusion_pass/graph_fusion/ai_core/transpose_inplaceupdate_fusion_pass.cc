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
 * \file transpose_inplaceupdate_fusion_pass.cpp
 * \brief transpose inplaceupdate fusion pass
 */
#include "transpose_inplaceupdate_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const char* TRANSPOSE = "TransposeD";
static const char* INPLACEUPDATE = "InplaceUpdate";
static const std::string PATTERN_TRANSPOSE0 = "FusedNodeTranspose0";
static const std::string PATTERN_INPLACEUPDATE0 = "FusedNodeInplaceUpdate0";
static const std::string PATTERN_TRANSPOSE1 = "FusedNodeTranspose1";
static const std::string PATTERN_INPUT0 = "Input0";

vector<FusionPattern*> TransposeInplaceUpdateFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransposeInplaceUpdateFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_TRANSPOSE0, {TRANSPOSE})
      .AddOpDesc(PATTERN_INPLACEUPDATE0, {INPLACEUPDATE})
      .AddOpDesc(PATTERN_TRANSPOSE1, {TRANSPOSE})
      .AddOpDesc(PATTERN_INPUT0)
      .SetInputs(PATTERN_TRANSPOSE0, {PATTERN_INPUT0})
      .SetInputs(PATTERN_INPLACEUPDATE0, {PATTERN_TRANSPOSE0})
      .SetInputs(PATTERN_TRANSPOSE1, {PATTERN_INPLACEUPDATE0})
      .SetOutput(PATTERN_TRANSPOSE1);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransposeInplaceUpdateFusionPass pattern end");
  return patterns;
}

Status TransposeInplaceUpdateFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                vector<ge::NodePtr>& fusionNodes) {
  // get all nodes
  ge::NodePtr trans0_node = GetNodeFromMapping(PATTERN_TRANSPOSE0, mapping);
  ge::NodePtr inplace0_node = GetNodeFromMapping(PATTERN_INPLACEUPDATE0, mapping);
  ge::NodePtr trans1_node = GetNodeFromMapping(PATTERN_TRANSPOSE1, mapping);
  FUSION_PASS_CHECK(trans0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transpose node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(inplace0_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inplaceupdate node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(trans1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transpose is null, fusion failed."),
                    return PARAM_INVALID);

  // get input
  ge::OpDescPtr trans0_desc = trans0_node->GetOpDesc();
  ge::OpDescPtr inplace0_desc = inplace0_node->GetOpDesc();
  ge::OpDescPtr trans1_desc = trans1_node->GetOpDesc();
  FUSION_PASS_CHECK(trans0_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transpose OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(inplace0_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inplaceupdate OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(trans1_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transpose OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc input_desc = trans0_desc->GetInputDesc(0);
  ge::GeTensorDesc output_desc = trans1_desc->GetOutputDesc(0);
  std::vector<int64_t> input_dims = input_desc.GetShape().GetDims();
  std::vector<int64_t> inplace_dims0 = inplace0_desc->GetInputDesc(0).GetShape().GetDims();
  std::vector<int64_t> inplace_dims1 = inplace0_desc->GetInputDesc(1).GetShape().GetDims();
  std::vector<int64_t> inplace_dims2 = inplace0_desc->GetInputDesc(2).GetShape().GetDims();

  if (inplace_dims0.size() < 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the first inplace shape size should not be less than 4, not changed.");
    return NOT_CHANGED;
  }

  for (size_t i = 1; i <= 3; i++) {
    auto dim = inplace_dims0[i];
    if (PatternFusionUtil::IsUnknownShape(dim)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ZTransposeInplaceUpdateFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  }

  if (PatternFusionUtil::IsUnknownShape(inplace_dims1[0]) ||
      PatternFusionUtil::IsUnknownShape(inplace_dims2[0])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ZTransposeInplaceUpdateFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  // get attr
  Operator op_trans0 = ge::OpDescUtils::CreateOperatorFromNode(trans0_node);
  Operator op_trans1 = ge::OpDescUtils::CreateOperatorFromNode(trans1_node);
  std::vector<int64_t> perm0;
  if (ge::GRAPH_SUCCESS != op_trans0.GetAttr("perm", perm0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr perm failed.");
    return NOT_CHANGED;
  }
  std::vector<int64_t> perm1;
  if (ge::GRAPH_SUCCESS != op_trans1.GetAttr("perm", perm1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr perm failed.");
    return NOT_CHANGED;
  }

  // verify
  if ((input_dims.size() == 4) && (perm0.size() == 4) && (perm1.size() == 4)) {
    if ((perm0[0] != 2) || (perm0[1] != 0) || (perm0[2] != 1) || (perm0[3] != 3)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule of transpose is not satisfied, not changed.");
      return NOT_CHANGED;
    }
    if ((perm1[0] != 1) || (perm1[1] != 2) || (perm1[2] != 0) || (perm1[3] != 3)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule of transpose is not satisfied, not changed.");
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the size of input is not four, not changed.");
    return NOT_CHANGED;
  }

  if ((inplace_dims0.size() != 4) || (inplace_dims1.size() != 1) || (inplace_dims2.size() != 4)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule1 of inplaceupdate is not satisfied, not changed.");
    return NOT_CHANGED;
  }

  if ((inplace_dims1[0] != 1) || (inplace_dims2[0] != 1)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule2 of inplaceupdate is not satisfied, not changed.");
    return NOT_CHANGED;
  }

  if ((inplace_dims0[1] != inplace_dims2[1]) || (inplace_dims0[2] != inplace_dims2[2]) ||
      (inplace_dims0[3] != inplace_dims2[3])) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule3 of inplaceupdate is not satisfied, not changed.");
    return NOT_CHANGED;
  }

  if ((inplace_dims0[3] % 16 != 0) || (inplace_dims0[3] > 256)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the rule4 of inplaceupdate is not satisfied, not changed.");
    return NOT_CHANGED;
  }

  // set input and output desc
  FUSION_PASS_CHECK(inplace0_desc->UpdateInputDesc(0, input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update inplaceupdate input desc failed."), return FAILED);
  FUSION_PASS_CHECK(inplace0_desc->UpdateOutputDesc(0, output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update inplaceupdate output desc failed."), return FAILED);

  // connect input edge
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(trans0_node->GetOutDataAnchor(0), inplace0_node->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(trans0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            inplace0_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            trans0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            inplace0_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto inDataAnchor : trans1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(trans1_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(inplace0_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(trans0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transpose node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(trans1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transpose node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "TransposeInplaceUpdateFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("ZTransposeInplaceUpdateFusionPass", BUILT_IN_GRAPH_PASS, TransposeInplaceUpdateFusionPass);
}  // namespace fe
