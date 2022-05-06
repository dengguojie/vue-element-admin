/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
* \file attention_qkv_gradx_fusion_pass.cc
* \brief the pass will turn three conjuction matmul into a big kernel
*  *  * pattern:
*                              layernormXBp
*       matmul_dx_query  ->         |
*       matmul_dx_key    ->       add_n     =>   AttentionQKVGradX
*       matmul_dx_value  ->
*/

#include <string>
#include "attention_qkv_gradx_fusion_pass.h"
#include "anchor_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const size_t kKernelNum = 3;
static const size_t kAddInputNum = 4;
static const size_t kMatmulInputNum = 2;
static const size_t kC0 = 16;
static const size_t kCoreNum32 = 32;
static const int64_t kCandidateM1 = 12288;
static const int64_t kCandidateN = 1024;
static const string kBoolToStr[2] = {"false", "true"};
static const string kPatternAddN = "AddN";
static const string kPatternLayerNormXBackprop = "LayerNormXBp";
static const string kOpMatmul = "MatMulV2";
static const string kOpAddn = "AddN";
static const string kOpLayernormXBackprop = "LayerNormXBackpropV2";
static const string kOpAttentionQKVGradX = "AttentionQKVGradX";
vector<FusionPattern *> AttentionQKVGradXFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("AttentionQKVGradXFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern->AddOpDesc(kPatternLayerNormXBackprop, {kOpLayernormXBackprop})
           .AddOpDesc(kPatternAddN, {kOpAddn})
           .SetInputs(kPatternAddN, {kPatternLayerNormXBackprop})
           .SetOutput(kPatternAddN);
  patterns.push_back(pattern);
  return patterns;
}

Status AttentionQKVGradXFusionPass::Fusion(ge::ComputeGraph &graph,
                                           Mapping &mapping,
                                           vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start AttentionQKVGradXFusionPass.");
  ge::NodePtr addn_node = GetNodeFromMapping(kPatternAddN, mapping);
  FUSION_PASS_CHECK(addn_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "addn_node is null, fusion failed."),
                    return PARAM_INVALID);
  std::vector<ge::NodePtr> matmul_list;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "addn_node is [%s].", addn_node->GetName().c_str());

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE, "Fail to get platform info.");
    optional_info.soc_version == "";
  }
  size_t core_num = platform_info.soc_info.ai_core_cnt;
  if (optional_info.soc_version.find("Ascend910") == string::npos || core_num != kCoreNum32) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "platform not supported.");
    return NOT_CHANGED;
  }

  if (!IsMatch(addn_node, matmul_list)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Match AttentionQKVGradXFusionPass failed.");
    return NOT_CHANGED;
  }
  ge::NodePtr attention_qkv_gradx_node = nullptr;
  FUSION_PASS_CHECK(SUCCESS != ReplaceAttentionQKVGradX(graph, addn_node, matmul_list, attention_qkv_gradx_node),
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReplaceAttentionQKVGradX failed!"), return FAILED);
  matmul_list.push_back(addn_node);
  for (auto &node : matmul_list) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGW(FUSED_OP_TYPE.c_str(),
                      "remove [%s] node failed.", node->GetName().c_str()), return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End AttentionQKVGradXFusionPass.");
  fusion_nodes.push_back(attention_qkv_gradx_node);
  return SUCCESS;
}

bool AttentionQKVGradXFusionPass::IsMatch(const ge::NodePtr &addn_node, std::vector<ge::NodePtr> &matmul_list) const {
  auto addn_in_anchors = addn_node->GetAllInDataAnchors();
  // pattern check
  if (addn_in_anchors.size() != kAddInputNum) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "input nums of addN unmatched.");
    return false;
  }
  for (size_t i = 1; i < kAddInputNum; i++) {
    auto addn_input_node = addn_node->GetInDataNodes().at(i);
    if (kOpMatmul != addn_input_node->GetType()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "input of addN is not matmulv2, match failed.");
      return false;
    }
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(addn_input_node->GetOpDesc(), "transpose_x1", transpose_x1),
        OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr transpose_x1 failed."), return false);
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(addn_input_node->GetOpDesc(), "transpose_x2", transpose_x2),
        OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr transpose_x2 failed."), return false);
    if (transpose_x1 || !transpose_x2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "trans_a/trans_b should be false/true, the actual trans_flags are [%s] and [%s].",
          kBoolToStr[transpose_x1], kBoolToStr[transpose_x2]);
      return false;
    }
    matmul_list.push_back(addn_input_node);
  }
  // shape check
  vector<int64_t> addn_out_shape = addn_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  bool invalid_out_shape = addn_out_shape[0] != kCandidateM1 || addn_out_shape[1] != kCandidateN;
  if (invalid_out_shape) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "invalid addn_out_shape.");
    return false;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AttentionQKVGradXFusionPass match success");
  return true;
}

Status AttentionQKVGradXFusionPass::ReplaceAttentionQKVGradX(ge::ComputeGraph &graph,
                                                             const ge::NodePtr &addn_node,
                                                             const std::vector<ge::NodePtr> &matmul_list,
                                                             ge::NodePtr &new_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter ReplaceAttentionQKVGradX.");
  auto addn_op_desc = addn_node->GetOpDesc();
  ge::OpDescPtr attention_qkv_gradx_desc;
  FUSION_PASS_MAKE_SHARED((attention_qkv_gradx_desc = std::make_shared<ge::OpDesc>(
                          addn_op_desc->GetName() + "/attention_qkv_gradx", kOpAttentionQKVGradX)), return FAILED);
  // AddInputDesc
  FUSION_PASS_CHECK(attention_qkv_gradx_desc->AddInputDesc("ln_dx", addn_op_desc->GetInputDesc(0).Clone()) !=
      GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add input desc ln_dx to attention_qkv_gradx."),
      return FAILED);
  std::vector<std::string> in_desc_names = {"query_dx", "key_dw", "value_dw",
                                            "kernel_query", "kernel_key", "kernel_value"};
  for (size_t i = 0; i < kMatmulInputNum; i++) {
    for (size_t j = 0; j < matmul_list.size(); j++) {
      auto input_desc = matmul_list[j]->GetOpDesc()->GetInputDesc(i);
      FUSION_PASS_CHECK(attention_qkv_gradx_desc->AddInputDesc(in_desc_names[i * kKernelNum + j], input_desc.Clone()) !=
          GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add input desc [%s] to attention_qkv_gradx.",
          in_desc_names[i * kKernelNum + j]), return FAILED);
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddInputDesc Success.");
  // AddOutputDesc
  FUSION_PASS_CHECK(attention_qkv_gradx_desc->AddOutputDesc("dx", addn_op_desc->GetOutputDesc(0).Clone()) !=
      GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add output desc dx to attention_qkv_gradx."),
      return FAILED);
  // AddNode
  auto attention_qkv_gradx_node = graph.AddNode(attention_qkv_gradx_desc);
  FUSION_PASS_CHECK(attention_qkv_gradx_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "failed to add attention_qkv_gradx to graph."), return FAILED);
  new_node = attention_qkv_gradx_node;
  auto ln_x_bp_out_anchor = addn_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  if (ge::GraphUtils::AddEdge(ln_x_bp_out_anchor, attention_qkv_gradx_node->GetInDataAnchor(0)) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_qkv_gradx failed.",
        ln_x_bp_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  for (size_t k = 0; k < kMatmulInputNum; k++) {
    for (size_t j = 0; j < matmul_list.size(); j++) {
      auto matmul_node = matmul_list[j];
      auto pre_out_anchor = matmul_node->GetInDataAnchor(k)->GetPeerOutAnchor();
      // RemoveEdge from input_node of matmul to matmul
      if (ge::GraphUtils::RemoveEdge(pre_out_anchor, matmul_node->GetInDataAnchor(k)) != SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from [%s] to matmul failed.",
            pre_out_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      // AddEdge from input_node of matmul to attention_qkv_gradx
      size_t index = k * kKernelNum + j + 1;
      if (ge::GraphUtils::AddEdge(pre_out_anchor, attention_qkv_gradx_node->GetInDataAnchor(index)) != SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_qkv_gradx_node failed.",
            pre_out_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add All Input Edge Success.");
  // handle addn output
  auto addn_out_anchor = addn_node->GetOutDataAnchor(0);
  auto next_in_anchors = addn_out_anchor->GetPeerInDataAnchors();
  for (auto next_in_anchor : next_in_anchors) {
    // RemoveEdge from addn_node to its output
    if (ge::GraphUtils::RemoveEdge(addn_out_anchor, next_in_anchor) != SUCCESS) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from addn to [%s] failed.",
          next_in_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
    // AddEdge from attention_qkv_gradx to addn_node's output
    if (ge::GraphUtils::AddEdge(attention_qkv_gradx_node->GetOutDataAnchor(0), next_in_anchor) != SUCCESS) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_qkv_gradx to [%s] failed.",
          next_in_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add All Output Edges Success.");
  return SUCCESS;
}

REGISTER_PASS("ZAttentionQKVGradXFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AttentionQKVGradXFusionPass);
} // namespace fe
