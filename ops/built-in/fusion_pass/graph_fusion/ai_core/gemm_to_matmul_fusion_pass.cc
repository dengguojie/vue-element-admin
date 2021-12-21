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
 * \file gemm_to_matmul_fusion_pass.cpp
 * \brief gemm to matmul+mul+add fusion pass
 */
#include "gemm_to_matmul_fusion_pass.h"

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
const string GemmToMatmulFusionPass::FUSED_OP_TYPE = "GEMM";

namespace {
static const char GEMM[] = "GEMM";
static const char PATTERN_GEMM[] = "GEMM";
}  // namespace

vector<FusionPattern*> GemmToMatmulFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmToMatmulFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("GemmToMatmulFusionPass");
  FUSION_PASS_CHECK(
    pattern == nullptr,
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
    return patterns
  );

  pattern->AddOpDesc(PATTERN_GEMM, {GEMM}).SetOutput(PATTERN_GEMM);
  patterns.push_back(pattern);
  return patterns;
}


Status GemmToMatmulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                       vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmToMatmulFusionPass.");

  // check whech gemm change to matmul
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
    PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Get platform_info failed."),
    return FAILED
  );
  map<string, vector<string>> intrinsic_map = platform_info.ai_core_intrinsic_dtype_map;
  if (intrinsic_map.size() == 0 || intrinsic_map.find("Intrinsic_fix_pipe_l0c2out") == intrinsic_map.end()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "this version no need change gemm to matmul");
    return NOT_CHANGED;
  }
  ge::NodePtr gemm_node = GetNodeFromMapping(PATTERN_GEMM, mapping);
  FUSION_PASS_CHECK(
    gemm_node == nullptr,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "gemm_node is null, fusion failed."),
    return PARAM_INVALID
  );

  int a_anchor = 0;
  int b_anchor = 1;
  int c_anchor = 2;
  int alpha_anchor = 3;
  int beta_anchor = 4;
  int y_anchor = 0;

  // get transpose flag
  bool transpose_a = false;
  bool transpose_b = false;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gemm_node);
  if (op.GetAttr("transpose_a", transpose_a) != GRAPH_SUCCESS or
      op.GetAttr("transpose_b", transpose_b) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "op gemm get transflag failed");
  }

  // get out/in anchor of input/out node, and the input/output node
  auto a_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, a_anchor);
  auto b_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, b_anchor);
  auto c_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, c_anchor);
  auto alpha_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, alpha_anchor);
  auto beta_anchor_ptr = GetPeerOutAnchorWithInDataAnchor(gemm_node, beta_anchor);
  auto gemm_out_ptr = gemm_node->GetOutDataAnchor(y_anchor);
  auto y_anchor_ptr = GetPeerInAnchorByOutDataAnchor(gemm_out_ptr, 0);
  FUSION_PASS_CHECK(
    a_anchor_ptr == nullptr || b_anchor_ptr == nullptr || c_anchor_ptr == nullptr ||
    alpha_anchor_ptr == nullptr || beta_anchor_ptr == nullptr || y_anchor_ptr == nullptr,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "anchor is null, fusion failed."),
    return PARAM_INVALID
  );
  vector<InDataAnchorPtr> y_anchors_ptr;
  for (auto in_data_anchor : gemm_out_ptr->GetPeerInDataAnchors()) {
    y_anchors_ptr.push_back(in_data_anchor);
  }

  auto a_node = a_anchor_ptr->GetOwnerNode();
  auto b_node = b_anchor_ptr->GetOwnerNode();
  auto c_node = c_anchor_ptr->GetOwnerNode();
  auto alpha_node = alpha_anchor_ptr->GetOwnerNode();
  auto beta_node = beta_anchor_ptr->GetOwnerNode();
  auto y_node = y_anchor_ptr->GetOwnerNode();
  FUSION_PASS_CHECK(
    a_node == nullptr || b_node == nullptr || c_node == nullptr ||
    alpha_node == nullptr || beta_node == nullptr || y_node == nullptr,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "input or output is null, fusion failed."),
    return PARAM_INVALID
  );

  int a_idx = a_anchor_ptr->GetIdx();
  int b_idx = b_anchor_ptr->GetIdx();
  int c_idx = c_anchor_ptr->GetIdx();
  int alpha_idx = alpha_anchor_ptr->GetIdx();
  int beta_idx = beta_anchor_ptr->GetIdx();

  // get desc of gemm
  ge::GeTensorDesc gemm_a_in_desc = gemm_node->GetOpDesc()->GetInputDesc(a_anchor);
  ge::GeTensorDesc gemm_b_in_desc = gemm_node->GetOpDesc()->GetInputDesc(b_anchor);
  ge::GeTensorDesc gemm_c_in_desc = gemm_node->GetOpDesc()->GetInputDesc(c_anchor);
  ge::GeTensorDesc gemm_alpha_in_desc = gemm_node->GetOpDesc()->GetInputDesc(alpha_anchor);
  ge::GeTensorDesc gemm_beta_in_desc = gemm_node->GetOpDesc()->GetInputDesc(beta_anchor);
  ge::GeTensorDesc gemm_out_desc = gemm_node->GetOpDesc()->GetOutputDesc(y_anchor);

  // matmul, mul1, mul2, add node
  ge::NodePtr matmul_node = nullptr;
  ge::NodePtr matmul_mul_node = nullptr;
  ge::NodePtr c_mul_node = nullptr;
  ge::NodePtr add_node = nullptr;
  fusion_nodes.clear();
  // matmul node
  ge::OpDescPtr matmul_desc;
  FUSION_PASS_MAKE_SHARED(
    matmul_desc = std::make_shared<ge::OpDesc>("gemm_to_matmul", "MatMulV2"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_desc->AddInputDesc("x1", gemm_a_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to matmul x1 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_desc->AddInputDesc("x2", gemm_b_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to matmul x2 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_desc->AddOutputDesc("y", gemm_out_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add out desc to matmul failed"),
    return FAILED
  );
  ge::AttrUtils::SetBool(matmul_desc, "transpose_x1", transpose_a);
  ge::AttrUtils::SetBool(matmul_desc, "transpose_x2", transpose_b);
  matmul_node = graph.AddNode(matmul_desc);
  fusion_nodes.push_back(matmul_node);
  // mul1 node
  ge::OpDescPtr matmul_mul_desc;
  FUSION_PASS_MAKE_SHARED(
    matmul_mul_desc = std::make_shared<ge::OpDesc>("matmul_mul1_1", "Mul"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_mul_desc->AddInputDesc("x1", gemm_out_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to mul1 x1 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_mul_desc->AddInputDesc("x2", gemm_alpha_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to mul1 x2 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    matmul_mul_desc->AddOutputDesc("y", gemm_out_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add out desc to mul1 failed"),
    return FAILED
  );
  matmul_mul_node = graph.AddNode(matmul_mul_desc);
  fusion_nodes.push_back(matmul_mul_node);
  // mul2 node
  ge::OpDescPtr c_mul_desc;
  FUSION_PASS_MAKE_SHARED(
    c_mul_desc = std::make_shared<ge::OpDesc>("matmul_mul1_2", "Mul"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    c_mul_desc->AddInputDesc("x1", gemm_c_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to mul2 x1 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    c_mul_desc->AddInputDesc("x2", gemm_beta_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to mul2 x2 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    c_mul_desc->AddOutputDesc("y", gemm_c_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add out desc to mul2 failed"),
    return FAILED
  );
  c_mul_node = graph.AddNode(c_mul_desc);
  fusion_nodes.push_back(c_mul_node);
  // add node
  ge::OpDescPtr add_desc;
  FUSION_PASS_MAKE_SHARED(
    add_desc = std::make_shared<ge::OpDesc>("matmul_add", "Add"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_desc->AddInputDesc("x1", gemm_out_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to add x1 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_desc->AddInputDesc("x2", gemm_c_in_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add input desc to add x2 failed"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_desc->AddOutputDesc("y", gemm_out_desc) != GRAPH_SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "add out desc to add failed"),
    return FAILED
  );
  add_node = graph.AddNode(add_desc);
  fusion_nodes.push_back(add_node);

  // remove the node with gemm_node
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(a_node->GetOutDataAnchor(a_idx), gemm_node->GetInDataAnchor(a_anchor)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between a_node and gemm_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(b_node->GetOutDataAnchor(b_idx), gemm_node->GetInDataAnchor(b_anchor)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between b_node and gemm_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(c_node->GetOutDataAnchor(c_idx), gemm_node->GetInDataAnchor(c_anchor)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between a_node and gemm_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(alpha_node->GetOutDataAnchor(alpha_idx),
                               gemm_node->GetInDataAnchor(alpha_anchor)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between alpha_node and gemm_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(beta_node->GetOutDataAnchor(beta_idx),
                               gemm_node->GetInDataAnchor(beta_anchor)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between beta_node and gemm_node"),
    return FAILED
  );
  for (auto input_data_anchor : y_anchors_ptr) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(gemm_node->GetOutDataAnchor(y_anchor), input_data_anchor) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between gemm_node and y_node"),
      return FAILED
    );
  }

  // add node with matmul, mul, add
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(a_node->GetOutDataAnchor(a_idx), matmul_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between a_node and matmul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(b_node->GetOutDataAnchor(b_idx), matmul_node->GetInDataAnchor(1)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between b_node and matmul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), matmul_mul_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between matmul_node and matmul_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(alpha_node->GetOutDataAnchor(alpha_idx), matmul_mul_node->GetInDataAnchor(1)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between matmul_node and matmul_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(c_node->GetOutDataAnchor(c_idx), c_mul_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between c_node and c_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(beta_node->GetOutDataAnchor(beta_idx), c_mul_node->GetInDataAnchor(1)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between beta_node and c_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(matmul_mul_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between matmul_mul_node and add_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(c_mul_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between c_mul_node and add_node"),
    return FAILED
  );
  for (auto input_data_anchor : y_anchors_ptr) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), input_data_anchor) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add the edge between add_node and y_node"),
      return FAILED
    );
  }

  // RemoveNode(gemm_node)
  FUSION_PASS_CHECK(
    graph.RemoveNode(gemm_node),
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove gemm_node"),
    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "remove gemm_node");
  return SUCCESS;
}

REGISTER_PASS("GemmToMatmulFusionPass", BUILT_IN_GRAPH_PASS, GemmToMatmulFusionPass);
}  // namespace fe
