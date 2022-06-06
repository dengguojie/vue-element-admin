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
 * \file gemm_transpose_fusion_pass.cpp
 * \brief gemm transpose fusion pass
 */
#include "gemm_transpose_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
const string GemmTransFusionPass::FUSED_OP_TYPE = "GEMM";

namespace {
static const char GEMM[] = "GEMM";
static const char PATTERN_GEMM[] = "GEMM";
static const int ALIGN_LENGTH = 16;
static const int kNumTwo = 2;
}  // namespace

vector<FusionPattern*> GemmTransFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("GemmTransFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_GEMM, {GEMM}).SetOutput(PATTERN_GEMM);

  patterns.push_back(pattern);

  return patterns;
}

Status GemmTransFusionPass::GenerateTransposeNode(ge::ComputeGraph* graph, const ge::GeTensorDesc& prev_out_desc,
                                                  ge::GeTensorDesc* next_in_desc, const vector<int64_t>& perm,
                                                  ge::NodePtr* transpose_node, const std::string& basename) {
  FUSION_PASS_CHECK(next_in_desc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "The next_in_desc is null."),
                    return FAILED);
  FUSION_PASS_CHECK(perm.size() > 2, OP_LOGW(FUSED_OP_TYPE.c_str(), "The perm size is larger than 2."),
                    return FAILED);
  vector<int64_t> next_in_shape(kNumTwo);
  for (size_t i = 0; i < perm.size(); ++i) {
    next_in_shape[i] = prev_out_desc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transpose_desc;
  FUSION_PASS_MAKE_SHARED((transpose_desc = std::make_shared<ge::OpDesc>(basename + "_transpose", "TransposeD")),
                          return FAILED);
  FUSION_PASS_CHECK(transpose_desc->AddInputDesc("x", prev_out_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add input desc to transpose node."),
                    return FAILED);
  next_in_desc->SetShape(ge::GeShape(next_in_shape));
  next_in_desc->SetOriginShape(ge::GeShape(next_in_shape));

  FUSION_PASS_CHECK(transpose_desc->AddOutputDesc("y", *next_in_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add output desc to transpose node."),
                    return FAILED);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", perm);

  auto new_transpose_node = graph->AddNode(transpose_desc);
  FUSION_PASS_CHECK(new_transpose_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add transpose node to graph."),
                    return FAILED);

  *transpose_node = new_transpose_node;
  return SUCCESS;
}

Status GemmTransFusionPass::Relink(ge::NodePtr a_node,
                                   ge::NodePtr transpose_a_node,
                                   ge::NodePtr gemm_node, const int anchor) {
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(a_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(anchor)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove edge between a_node and gemm_node."),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), transpose_a_node->GetInDataAnchor(0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge between a_node and transpose_a_node."),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transpose_a_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(anchor)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge between transpose_a_node and gemm_node."),
      return FAILED);

  FUSION_PASS_CHECK(
      gemm_node->GetOpDesc()->UpdateInputDesc(anchor, transpose_a_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to update input description of transdataANode."),
      return FAILED);

  return SUCCESS;
}

Status GemmTransFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                   vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr gemm_node = GetNodeFromMapping(PATTERN_GEMM, mapping);
  FUSION_PASS_CHECK(gemm_node == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because gemm node is null."),
                    return NOT_CHANGED);

  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get platform_info."), return NOT_CHANGED);
  map<string, vector<string>> intrinsic_map = platform_info.ai_core_intrinsic_dtype_map;
  FUSION_PASS_CHECK(intrinsic_map.size() > 0 && intrinsic_map.find("Intrinsic_fix_pipe_l0c2out") != intrinsic_map.end(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Gemm node will change to matmul in this soc version."),
                    return NOT_CHANGED);

  int a_anchor = 0;
  int b_anchor = 1;
  int c_anchor = 2;

  // get transpose flag
  bool transpose_a = false;
  bool transpose_b = false;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gemm_node);
  if (op.GetAttr("transpose_a", transpose_a) != GRAPH_SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get gemm's attribute [transpose_a].");
  }

  if (op.GetAttr("transpose_b", transpose_b) != GRAPH_SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get gemm's attribute [transpose_b].");
  }

  // prerequisite
  auto a_node = GetPeerOutNodeWithInDataAnchor(gemm_node, a_anchor);
  FUSION_PASS_CHECK(a_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because a is null."),
                    return NOT_CHANGED);
  auto a_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, a_anchor);
  FUSION_PASS_CHECK(a_anchor_ptr == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because anchor a is null."),
                    return NOT_CHANGED);
  int a_idx = a_anchor_ptr->GetIdx();

  auto b_node = GetPeerOutNodeWithInDataAnchor(gemm_node, b_anchor);
  FUSION_PASS_CHECK(b_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because b is null."),
                    return NOT_CHANGED);
  auto b_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, b_anchor);
  FUSION_PASS_CHECK(b_anchor_ptr == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because anchor b is null."),
                    return NOT_CHANGED);
  int b_idx = b_anchor_ptr->GetIdx();

  ge::NodePtr c_node = GetPeerOutNodeWithInDataAnchor(gemm_node, c_anchor);
  FUSION_PASS_CHECK(c_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because c is null."),
                    return NOT_CHANGED);
  auto c_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, c_anchor);
  FUSION_PASS_CHECK(c_anchor_ptr == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to do fusion because anchor c is null."),
                    return NOT_CHANGED);
  int c_idx = c_anchor_ptr->GetIdx();

  // get info of Node
  ge::GeTensorDesc a_out_desc = a_node->GetOpDesc()->GetOutputDesc(a_idx);
  ge::GeTensorDesc b_out_desc = b_node->GetOpDesc()->GetOutputDesc(b_idx);
  ge::GeTensorDesc c_out_desc = c_node->GetOpDesc()->GetOutputDesc(c_idx);

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
  FUSION_PASS_CHECK(b_shape_vector.size() < 2, OP_LOGW(FUSED_OP_TYPE.c_str(), "The size of b_shape is less than 2."),
                    return NOT_CHANGED);
  if (transpose_b) {
    n_direction_length = b_shape_vector[0];
  } else {
    n_direction_length = b_shape_vector[1];
  }

  FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(n_direction_length),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "GemmTransFusionPass cannot be applied for unknown shape."),
                    return NOT_CHANGED);

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
    FUSION_PASS_CHECK(GenerateTransposeNode(&graph, a_out_desc, &gemm_a_in_desc, trans_perm, &transpose_a_node,
                                            basename_a) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to generate transpose node A."),
                      return FAILED);
    // relink a
    FUSION_PASS_CHECK(Relink(a_node, transpose_a_node, gemm_node, a_anchor) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to relink nodes A."), return FAILED);
    fusion_nodes.push_back(transpose_a_node);
    op.SetAttr("transpose_a", false);
  }

  if (transpose_b && need_transpose) {
    // transpose b
    FUSION_PASS_CHECK(GenerateTransposeNode(&graph, b_out_desc, &gemm_b_in_desc, trans_perm, &transpose_b_node,
                                            basename_b) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to generate transpose node B."),
                      return FAILED);
    // relink b
    FUSION_PASS_CHECK(Relink(b_node, transpose_b_node, gemm_node, b_anchor) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to relink nodes B."), return FAILED);
    fusion_nodes.push_back(transpose_b_node);
    op.SetAttr("transpose_b", false);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Do GemmTransFusionPass success.");
  return SUCCESS;
}

REGISTER_PASS("GemmTransFusionPass", BUILT_IN_GRAPH_PASS, GemmTransFusionPass);
}  // namespace fe
