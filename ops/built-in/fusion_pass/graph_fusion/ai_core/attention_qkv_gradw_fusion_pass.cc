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
* \file attention_qkv_gradw_fusion_pass.cc
* \brief the pass will turn three conjuction matmul into a big kernel
*/

#include "attention_qkv_gradw_fusion_pass.h"
#include "anchor_util.h"
#include "graph/utils/graph_utils.h"
#include "common/util/platform_info.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const size_t kKernelNum = 3;
static const size_t kReduceSumIdx = 2;
static const int64_t kCandidateK = 12288;
static const int64_t kCandidateM = 1024;
static const int64_t kMaxShape = 65536;
static const string kBoolToStr[2] = {"false", "true"};
static const string kPatternBatchMatmul = "BatchMatMul";
static const string kPatternConfTranspose = "ConfTranspose";
static const string kPatternReduceSum = "ReduceSum";
static const string kPatternHeadNode = "HeadNode";
static const string kOpDropoutDoMaskV3 = "DropOutDoMaskV3D";
static const string kOpBatchMatmul = "BatchMatMulV2";
static const string kOpFastGeluGrad = "FastGeluGrad";
static const string kOpLNDropoutGrad = "LNDropoutGrad";
static const string kOpMatMul = "MatMulV2";
static const string kOpConftransposeD = "ConfusionTransposeD";
static const string kOpReduceSumD = "ReduceSumD";
static const string kOpAttentionQKVGradW = "AttentionQKVGradW";
vector<FusionPattern *> AttentionQKVGradWFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("AttentionQKVGradWFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern->AddOpDesc(kPatternBatchMatmul, {kOpBatchMatmul})
           .AddOpDesc(kPatternConfTranspose, {kOpConftransposeD})
           .AddOpDesc(kPatternReduceSum, {kOpReduceSumD})
           .SetInputs(kPatternConfTranspose, {kPatternBatchMatmul})
           .SetInputs(kPatternReduceSum, {kPatternConfTranspose})
           .SetOutput(kPatternReduceSum);
  patterns.push_back(pattern);
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("AttentionQKVGradWFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern1->AddOpDesc(kPatternHeadNode, {kOpDropoutDoMaskV3, kOpFastGeluGrad, kOpLNDropoutGrad})
           .AddOpDesc(kPatternReduceSum, {kOpReduceSumD})
           .SetInputs(kPatternReduceSum, {kPatternHeadNode})
           .SetOutput(kPatternReduceSum);
  patterns.push_back(pattern1);
  return patterns;
}

Status AttentionQKVGradWFusionPass::Fusion(ge::ComputeGraph &graph,
                                           Mapping &mapping,
                                           vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start AttentionQKVGradWFusionPass.");
  ge::NodePtr conf_transpose_node = GetNodeFromMapping(kPatternConfTranspose, mapping);
  ge::NodePtr reduce_sum_node = GetNodeFromMapping(kPatternReduceSum, mapping);
  FUSION_PASS_CHECK(reduce_sum_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "reduce_sum_node is null, fusion failed."), return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "reduce_sum_node is [%s].", reduce_sum_node->GetName().c_str());
  if (conf_transpose_node == nullptr) {
    single_mode = true;
    conf_transpose_node = reduce_sum_node->GetInDataNodes().at(0);
  }

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE, "Fail to get platform info.");
    optional_info.soc_version == "";
  }
  const auto &instrinsicMap = platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"];
  bool dtype_support_flag =
      find(instrinsicMap.begin(), instrinsicMap.end(), "deq") != instrinsicMap.end() &&
      find(instrinsicMap.begin(), instrinsicMap.end(), "f162s32a") != instrinsicMap.end();
  if (!dtype_support_flag) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "platform not supported.");
    return NOT_CHANGED;
  }

  std::vector<ge::NodePtr> matmul_list;
  std::vector<ge::NodePtr> reduce_sum_list;
  if (!IsMatch(conf_transpose_node, reduce_sum_node, matmul_list, reduce_sum_list)) {
    return NOT_CHANGED;
  }
  ge::NodePtr attention_qkv_gradw_node = nullptr;
  FUSION_PASS_CHECK(SUCCESS != ReplaceAttentionQKVGradW(graph, matmul_list, reduce_sum_list, attention_qkv_gradw_node),
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReplaceAttentionQKVGradW failed!"), return NOT_CHANGED);
  std::vector<ge::NodePtr> node_list;
  node_list.insert(node_list.end(), reduce_sum_list.begin(), reduce_sum_list.end());
  node_list.insert(node_list.end(), matmul_list.begin(), matmul_list.end());
  for (auto &node : node_list) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "remove [%s] node failed.", node->GetName().c_str()), return NOT_CHANGED);
  }
  OP_LOGD(attention_qkv_gradw_node, "End AttentionQKVGradWFusionPass.");
  fusion_nodes.push_back(attention_qkv_gradw_node);
  return SUCCESS;
}

bool AttentionQKVGradWFusionPass::IsMatch(const ge::NodePtr &conf_transpose_node,
                                          const ge::NodePtr &sum_node,
                                          std::vector<ge::NodePtr> &matmul_list,
                                          std::vector<ge::NodePtr> &reduce_sum_list) const {
  auto conf_trans_out_anchor = conf_transpose_node->GetOutDataAnchor(0);
  size_t matmul_dw_idx = 1;
  if (kOpLNDropoutGrad == conf_transpose_node->GetType()) {
    conf_trans_out_anchor = conf_transpose_node->GetOutDataAnchor(1);
    // matmul_dw_idx is 2 if the head node op_type is LNDropoutGrad
    matmul_dw_idx = 2;
  }
  // conf_transpose_node output is passed to matmul_dx, matmul_dw and reduce_sum_d
  if (conf_trans_out_anchor->GetPeerInDataAnchors().size() != kKernelNum) {
    OP_LOGD(conf_transpose_node, "output nums of head node [%zu] unmatched.",
        conf_trans_out_anchor->GetPeerInDataAnchors().size());
    return false;
  }
  auto matmul_dw_node = conf_transpose_node->GetOutDataNodes().at(matmul_dw_idx);
  auto out_anchor = matmul_dw_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  if (!single_mode && out_anchor->GetPeerInDataAnchors().size() < kKernelNum) {
    OP_LOGD(matmul_dw_node, "num of matmul_dw nodes [%zu] unmatched.", out_anchor->GetPeerInDataAnchors().size());
    return false;
  }
  size_t matmul_nums = kKernelNum;
  if (single_mode) {
    matmul_nums = 1;
  }
  for (size_t i = 0; i < matmul_nums; i++) {
    auto matmul_node = out_anchor->GetPeerInDataAnchors().at(i)->GetOwnerNode();
    if (single_mode) {
      matmul_node = matmul_dw_node;
    }
    if (kOpMatMul != matmul_node->GetType()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "matmul_node [%s] unmatched in node reversing.",
          matmul_node->GetName().c_str());
      return false;
    }
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(matmul_node->GetOpDesc(), "transpose_x1", transpose_x1),
        OP_LOGW(matmul_node, "get attr transpose_x1 failed."), return false);
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(matmul_node->GetOpDesc(), "transpose_x2", transpose_x2),
        OP_LOGW(matmul_node, "get attr transpose_x2 failed."), return false);
    if (!transpose_x1 || transpose_x2) {
      OP_LOGI(matmul_node, "trans_a/trans_b should be true/false, the actual trans_flags are [%s] and [%s].",
          kBoolToStr[transpose_x1].c_str(), kBoolToStr[transpose_x2].c_str());
      return false;
    }
    // recheck another input
    auto transpose_node = matmul_node->GetInDataNodes().at(1);
    if (!single_mode && kOpConftransposeD != transpose_node->GetType()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "conf_transpose_node [%s] type unmatched in node reversing.",
          transpose_node->GetName().c_str());
      return false;
    }
    auto reduce_sum_node = sum_node;
    if (!single_mode) {
      reduce_sum_node = transpose_node->GetOutDataNodes().at(kReduceSumIdx);
      if (kOpReduceSumD != reduce_sum_node->GetType()) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "reduce_sum node [%s] unmatched in node reversing.",
            reduce_sum_node->GetName().c_str());
        return false;
      }
    }
    matmul_list.push_back(matmul_node);
    reduce_sum_list.push_back(reduce_sum_node);
  }
  // check op info
  if (!CheckOpInfo(matmul_list[0], reduce_sum_list[0])) {
    return false;
  }
  return true;
}

bool AttentionQKVGradWFusionPass::CheckOpInfo(const ge::NodePtr &matmul_node,
                                              const ge::NodePtr &reduce_sum_node) const {
  // matmul shape check
  vector<int64_t> x_shape = matmul_node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape().GetDims();
  vector<int64_t> kernel_shape = matmul_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape().GetDims();
  bool invalid_matmul_shape = x_shape[0] != kCandidateK || x_shape[1] != kCandidateM || kernel_shape[1] != kCandidateM;
  if (single_mode) {
    invalid_matmul_shape = x_shape[0] != kCandidateK || x_shape[1] != kCandidateM ||
        kernel_shape[1] % kCandidateM != 0 || kernel_shape[1] > kMaxShape;
  }
  if (invalid_matmul_shape) {
    OP_LOGI(matmul_node, "The matmul input shape with m=%ld, k=%ld, n=%ld is not supported.",
        x_shape[1], x_shape[0], kernel_shape[1]);
    return false;
  }
  // reduce_sum shape check
  auto sum_out_desc = reduce_sum_node->GetOpDesc()->MutableOutputDesc(0);
  vector<int64_t> sum_out_shape = sum_out_desc->GetOriginShape().GetDims();
  if (sum_out_shape[0] != kernel_shape[1]) {
    OP_LOGI(matmul_node, "The reduce_sum out shape [%ld] is not supported.", sum_out_shape[0]);
    return false;
  }
  // dtype check
  auto matmul_out_desc = matmul_node->GetOpDesc()->MutableOutputDesc(0);
  if (matmul_out_desc->GetDataType() != ge::DT_FLOAT16 || sum_out_desc->GetDataType() != ge::DT_FLOAT16) {
    OP_LOGI(matmul_node, "matmul_node/reduce_sum_node dtype is not fp16, but [%s]/[%s]!",
        ge::TypeUtils::DataTypeToSerialString(matmul_out_desc->GetDataType()).c_str(),
        ge::TypeUtils::DataTypeToSerialString(sum_out_desc->GetDataType()).c_str());
    return false;
  }
  // format check
  if (matmul_out_desc->GetFormat() != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGI(matmul_node, "matmul_node output format is not FRACTAL_NZ, but [%s]!",
        ge::TypeUtils::FormatToSerialString(matmul_out_desc->GetFormat()).c_str());
    return false;
  }
  return true;
}

Status AttentionQKVGradWFusionPass::ReplaceAttentionQKVGradW(ge::ComputeGraph &graph,
                                                             const std::vector<ge::NodePtr> &matmul_list,
                                                             const std::vector<ge::NodePtr> &reduce_sum_list,
                                                             ge::NodePtr &new_node) {
  ge::OpDescPtr attention_qkv_gradw_desc;
  FUSION_PASS_MAKE_SHARED((attention_qkv_gradw_desc = std::make_shared<ge::OpDesc>(
                          matmul_list[0]->GetName() + "/attention_qkv_grad_w", kOpAttentionQKVGradW)), return FAILED);
  std::vector<std::string> input_names = {"query_dx", "key_dw", "value_dw"};
  std::vector<std::string> output_names = {"dw_query", "dw_key", "dw_value",
                                           "dbias_qurey", "dbias_key", "dbias_value"};
  // AddInputDesc
  FUSION_PASS_CHECK(attention_qkv_gradw_desc->AddInputDesc("x",
      matmul_list[0]->GetOpDesc()->MutableInputDesc(0)->Clone()) != GRAPH_SUCCESS,
      OP_LOGW(matmul_list[0], "failed to add input desc x to attention_qkv_grad_w."), return FAILED);
  for (size_t i = 0; i < matmul_list.size(); i++) {
    auto matmul_op_desc = matmul_list[i]->GetOpDesc();
    FUSION_PASS_CHECK(attention_qkv_gradw_desc->AddInputDesc(input_names[i],
        matmul_op_desc->MutableInputDesc(1)->Clone()) != GRAPH_SUCCESS,
        OP_LOGW(matmul_list[0], "failed to add input desc [%s] to attention_qkv_grad_w.", input_names[i].c_str()),
        return FAILED);
  }
  // AddOutputDesc
  vector<vector<ge::NodePtr>> list_out_nodes = {matmul_list, reduce_sum_list};
  for (size_t i = 0; i < list_out_nodes.size(); i++) {
    vector<ge::NodePtr> out_nodes = list_out_nodes[i];
    for (size_t j = 0; j < kKernelNum; j++) {
      auto op_desc = out_nodes[0]->GetOpDesc();
      FUSION_PASS_CHECK(attention_qkv_gradw_desc->AddOutputDesc(output_names[kKernelNum * i + j],
          op_desc->MutableOutputDesc(0)->Clone()) != GRAPH_SUCCESS, OP_LOGW(out_nodes[0],
          "failed to add output desc [%s] to attention_qkv_grad_w.", output_names[kKernelNum * i + j].c_str()),
          return FAILED);
    }
  }
  // AddNode
  auto attention_qkv_gradw_node = graph.AddNode(attention_qkv_gradw_desc);
  FUSION_PASS_CHECK(attention_qkv_gradw_node == nullptr, OP_LOGW(matmul_list[0],
      "failed to add attention_qkv_grad_w to graph."), return FAILED);
  new_node = attention_qkv_gradw_node;
  // Add Input Edge
  auto norm_out_anchor = matmul_list[0]->GetInDataAnchor(0)->GetPeerOutAnchor();
  if (ge::GraphUtils::AddEdge(norm_out_anchor, attention_qkv_gradw_node->GetInDataAnchor(0)) != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_qkv_grad_w failed.",
        norm_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  for (size_t i = 0; i < matmul_list.size(); i++) {
    auto matmul_node = matmul_list[i];
    auto sum_node = reduce_sum_list[i];
    // Add Input Edge
    auto dy_out_anchor = matmul_node->GetInDataAnchor(1)->GetPeerOutAnchor();
    if (ge::GraphUtils::AddEdge(dy_out_anchor, attention_qkv_gradw_node->GetInDataAnchor(i + 1)) != SUCCESS) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_qkv_grad_w failed.",
          dy_out_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
    // Remove/Add Output Edge
    auto matmul_out_anchor = matmul_node->GetOutDataAnchor(0);
    for (auto peer_in_anchor : matmul_out_anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(matmul_out_anchor, peer_in_anchor) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge from matmul to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      if (ge::GraphUtils::AddEdge(attention_qkv_gradw_node->GetOutDataAnchor(i), peer_in_anchor) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from attention_qkv_gradw_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
    auto sum_out_anchor = sum_node->GetOutDataAnchor(0);
    for (auto peer_in_anchor : sum_out_anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(sum_out_anchor, peer_in_anchor) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge from reduce_sum to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      if (ge::GraphUtils::AddEdge(attention_qkv_gradw_node->GetOutDataAnchor(i + kKernelNum), peer_in_anchor) !=
          SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from attention_qkv_gradw_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

REGISTER_PASS("ZAttentionQKVGradWFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AttentionQKVGradWFusionPass);
} // namespace fe
