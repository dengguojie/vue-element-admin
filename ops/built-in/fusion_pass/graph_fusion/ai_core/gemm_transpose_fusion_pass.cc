/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file gemm_transpose_fusion_pass.cpp
 * \brief gemm transpose fusion pass
 */
#include "gemm_transpose_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "error_util.h"

namespace fe {
namespace {
static const char GEMM[] = "GEMM";
static const char PATTERN_GEMM[] = "GEMM";
static const int ALIGN_LENGTH = 16;
}  // namespace

vector<FusionPattern*> GemmTransFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmTransFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("GemmTransFusionPass");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
      return patterns);

  pattern->AddOpDesc(PATTERN_GEMM, {GEMM}).SetOutput(PATTERN_GEMM);

  patterns.push_back(pattern);

  return patterns;
}

static Status GenerateTransposeNode(ge::ComputeGraph* graph,
                                    const ge::GeTensorDesc& prev_out_desc,
                                    ge::GeTensorDesc* next_in_desc,
                                    const vector<int64_t>& perm,
                                    ge::NodePtr* transpose_node,
                                    const std::string& basename) {
  vector<int64_t> next_in_shape(2);
  for (size_t i = 0; i < perm.size(); ++i) {
    next_in_shape[i] = prev_out_desc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transpose_desc;
  FUSION_PASS_MAKE_SHARED((transpose_desc = std::make_shared<ge::OpDesc>(
                               basename + "_transpose", "TransposeD")),
                          return FAILED);
  transpose_desc->AddInputDesc("x", prev_out_desc);
  next_in_desc->SetShape(ge::GeShape(next_in_shape));
  next_in_desc->SetOriginShape(ge::GeShape(next_in_shape));
  transpose_desc->AddOutputDesc("y", *next_in_desc);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", perm);
  *transpose_node = graph->AddNode(transpose_desc);
  return SUCCESS;
}

Status GemmTransFusionPass::Relink(ge::NodePtr a_node,
                                   ge::NodePtr transpose_a_node,
                                   ge::NodePtr gemm_node, const int anchor) {
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(a_node->GetOutDataAnchor(0),
                                 gemm_node->GetInDataAnchor(anchor)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
              "fail to remove edge between a_node and gemm_node"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(a_node->GetOutDataAnchor(0),
                              transpose_a_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
              "fail to add edge between a_node and transpose_a_node"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transpose_a_node->GetOutDataAnchor(0),
                              gemm_node->GetInDataAnchor(anchor)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
              "fail to add edge between transpose_a_node and gemm_node"),
      return FAILED);

  FUSION_PASS_CHECK(
      gemm_node->GetOpDesc()->UpdateInputDesc(
          anchor, transpose_a_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
              "fail to update input description of transdataANode"),
      return FAILED);

  return SUCCESS;
}

Status GemmTransFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                   vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmTransFusionPass.");
  ge::NodePtr gemm_node = GetNodeFromMapping(PATTERN_GEMM, mapping);
  FUSION_PASS_CHECK(gemm_node == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "gemm_node is null, fusion failed."),
                    return PARAM_INVALID);

  int a_anchor = 0;
  int b_anchor = 1;
  int c_anchor = 2;

  // get transpose flag
  bool transpose_a = false;
  bool transpose_b = false;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gemm_node);

  if (op.GetAttr("transpose_a", transpose_a) != GRAPH_SUCCESS) {
    OP_LOGI(
        FUSED_OP_TYPE.c_str(),
        "op gemm get attribute transpose_a failed or transpose_a not exist");
  }

  if (op.GetAttr("transpose_b", transpose_b) != GRAPH_SUCCESS) {
    OP_LOGI(
        FUSED_OP_TYPE.c_str(),
        "op gemm get attribute transpose_b failed or transpose_b not exist");
  }

  // prerequisite
  ge::NodePtr a_node =
      gemm_node->GetInDataAnchor(a_anchor)->GetPeerOutAnchor()->GetOwnerNode();
  int aIdx = gemm_node->GetInDataAnchor(a_anchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr b_node =
      gemm_node->GetInDataAnchor(b_anchor)->GetPeerOutAnchor()->GetOwnerNode();
  int bIdx = gemm_node->GetInDataAnchor(b_anchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr c_node =
      gemm_node->GetInDataAnchor(c_anchor)->GetPeerOutAnchor()->GetOwnerNode();
  int cIdx = gemm_node->GetInDataAnchor(c_anchor)->GetPeerOutAnchor()->GetIdx();

  // get info of Node
  ge::GeTensorDesc a_out_desc = a_node->GetOpDesc()->GetOutputDesc(aIdx);
  ge::GeTensorDesc b_out_desc = b_node->GetOpDesc()->GetOutputDesc(bIdx);
  ge::GeTensorDesc c_out_desc = c_node->GetOpDesc()->GetOutputDesc(cIdx);

  ge::GeTensorDesc gemm_a_in_desc =
      gemm_node->GetOpDesc()->GetInputDesc(a_anchor);
  ge::GeTensorDesc gemm_b_in_desc =
      gemm_node->GetOpDesc()->GetInputDesc(b_anchor);

  // get format and shape of a,b,c
  ge::Format a_format = a_out_desc.GetFormat();
  ge::Format b_format = b_out_desc.GetFormat();
  ge::Format c_format = c_out_desc.GetFormat();
  ge::GeShape a_shape = a_out_desc.GetShape();
  ge::GeShape b_shape = b_out_desc.GetShape();

  // get n_direction_length
  int n_direction_length = 0;
  std::vector<int64_t> b_shape_vector = b_shape.GetDims();
  FUSION_PASS_CHECK(b_shape_vector.size() < 2,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "b_shape is illegal"),
                    return FAILED);
  if (transpose_b) {
    n_direction_length = b_shape_vector[0];
  } else {
    n_direction_length = b_shape_vector[1];
  }

  if (PatternFusionUtil::IsUnknownShape(n_direction_length)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "GemmTransFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  bool need_transpose = true;

  if (a_format == ge::FORMAT_ND && b_format == ge::FORMAT_ND &&
      c_format == ge::FORMAT_ND && (n_direction_length % ALIGN_LENGTH == 0)) {
    need_transpose = false;
  }

  // 2. transpose
  ge::NodePtr transpose_a_node = nullptr;
  ge::NodePtr transpose_b_node = nullptr;
  auto basename_a = a_node->GetName();
  auto basename_b = b_node->GetName();
  vector<int64_t> trans_perm({1, 0});

  if (transpose_a && need_transpose) {
    // transpose a
    FUSION_PASS_CHECK(
        GenerateTransposeNode(&graph, a_out_desc, &gemm_a_in_desc, trans_perm,
                              &transpose_a_node, basename_a) != SUCCESS,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate transpose node A"),
        return FAILED);
    // relink a
    FUSION_PASS_CHECK(
        Relink(a_node, transpose_a_node, gemm_node, a_anchor) != SUCCESS,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);
    fusion_nodes.push_back(transpose_a_node);
    op.SetAttr("transpose_a", false);
  }

  if (transpose_b && need_transpose) {
    // transpose b
    FUSION_PASS_CHECK(
        GenerateTransposeNode(&graph, b_out_desc, &gemm_b_in_desc, trans_perm,
                              &transpose_b_node, basename_b) != SUCCESS,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate transpose node B"),
        return FAILED);
    // relink b
    FUSION_PASS_CHECK(
        Relink(b_node, transpose_b_node, gemm_node, b_anchor) != SUCCESS,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);
    fusion_nodes.push_back(transpose_b_node);
    op.SetAttr("transpose_b", false);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End GemmTransFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("GemmTransFusionPass", BUILT_IN_GRAPH_PASS, GemmTransFusionPass);
}  // namespace fe
