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
 * \file attention_ln_qkv_onnx_fusion_pass.cc
 * \brief the pass will turn three conjuction matmul_confusionTranspose into a attention_ln_qkv
 *  *  *  *Inference pattern:
 *                        LayerNorm
 *                      /     |     \
 *                    var     |     mean
 *                         /  |  \            =>   attention_ln_qkv
 *                        /   |   \
 *                   bmm_q  bmm_k  bmm_v
 *                     /      |     \
 *           ConTrans_q  ConTrans_k  ConTrans_v
 */
#include "attention_ln_qkv_onnx_fusion_pass.h"
#include "anchor_util.h"
#include "graph/utils/graph_utils.h"
#include "common/util/platform_info.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace {
  const int kKernelNum = 3;
  const int kXIndex = 0;
  const int kGammaIndex = 4;
  const int kBetaIndex = 5;
  const int kLnMeanIndex = 1;
  const int kLnVarIndex = 2;
  const int kBiasInputIndex = 6;
  const int kOutDimSize = 6;
  const int kMInnerIndex = 3;
  const int kInferOutNum = 4;
  const size_t kSupportedCoreNum1 = 8;
  const size_t kSupportedCoreNum2 = 32;
  const int64_t kC0 = 16;
  const int64_t kInferMinMShape1 = 1536;
  const int64_t kInferMinMShape2 = 2048;
  const int64_t kCandidateN1 = 1024;
  const int64_t kCandidateN2 = 768;
  const int64_t kCandidateTilingM1 = 12;
  const int64_t kCandidateTilingM2 = 8;
  const string kPatternLayernorm = "LayerNorm";
  const string kPatternBatchMatmul = "BatchMatMul";
  const string kPatternConfusionTranspose = "ConfusionTransposeD";
  const string kOpLayernorm = "LayerNorm";
  const string kOpBatchMatmulV2 = "BatchMatMulV2";
  const string kOpConfusionTranspose = "ConfusionTransposeD";
  const string kOpAttentionLnQKV = "AttentionLnQKV";
}

namespace fe {
vector<FusionPattern *> AttentionLnQKVONNXFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("AttentionLnQKVONNXFusionPass1");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern1->AddOpDesc(kPatternLayernorm, {kOpLayernorm})
           .AddOpDesc(kPatternBatchMatmul, {kOpBatchMatmulV2})
           .AddOpDesc(kPatternConfusionTranspose, {kOpConfusionTranspose})
           .SetInputs(kPatternBatchMatmul, {kPatternLayernorm})
           .SetInputs(kPatternConfusionTranspose, {kPatternBatchMatmul})
           .SetOutput(kPatternConfusionTranspose);
  patterns.push_back(pattern1);
  return patterns;
}

Status AttentionLnQKVONNXFusionPass::Fusion(ge::ComputeGraph &graph,
                                            Mapping &mapping,
                                            vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start AttentionLnQKVONNXFusionPass.");
  ge::NodePtr ln_node = GetNodeFromMapping(kPatternLayernorm, mapping);
  FUSION_PASS_CHECK(ln_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "layer_norm node is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ln_node is [%s].", ln_node->GetName().c_str());

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE, "Fail to get platform info.");
    optional_info.soc_version == "";
  }
  size_t core_num = platform_info.soc_info.ai_core_cnt;
  const auto &instrinsicMap = platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"];
  bool support_fp32_flag =
      find(instrinsicMap.begin(), instrinsicMap.end(), "float32") != instrinsicMap.end();
  if (!support_fp32_flag || (core_num != kSupportedCoreNum1 && core_num != kSupportedCoreNum2)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "platform not supported.");
    return NOT_CHANGED;
  }

  std::vector<ge::NodePtr> conf_trans_list;
  std::vector<ge::NodePtr> matmul_list;
  if (!IsMatch(ln_node, conf_trans_list, matmul_list)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Match AttentionLnQKVONNXFusionPass failed.");
    return NOT_CHANGED;
  }
  ge::NodePtr attention_ln_qkv_node = nullptr;
  FUSION_PASS_CHECK(SUCCESS != CreateAttentionLnQKVNode(graph, ln_node, conf_trans_list,
                                                        matmul_list, attention_ln_qkv_node),
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateAttentionLnQKVNode failed!"), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ReplaceAttentionLnQKV(graph, ln_node, conf_trans_list,
                                                     matmul_list, attention_ln_qkv_node),
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReplaceAttentionLnQKV failed!"), return FAILED);
  std::vector<ge::NodePtr> node_list;
  node_list.insert(node_list.end(), conf_trans_list.begin(), conf_trans_list.end());
  node_list.insert(node_list.end(), matmul_list.begin(), matmul_list.end());
  for (auto &node : node_list) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "remove [%s] node failed.", node->GetName().c_str()), return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End AttentionLnQKVONNXFusionPass.");
  fusion_nodes.push_back(attention_ln_qkv_node);
  return SUCCESS;
}

bool AttentionLnQKVONNXFusionPass::IsMatch(const ge::NodePtr &ln_node,
                                           std::vector<ge::NodePtr> &conf_trans_list,
                                           std::vector<ge::NodePtr> &matmul_list) {
  // pattern check
  auto out_anchor = ln_node->GetOutDataAnchor(0);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  // layernorm outputs are add node and three matmuls in inference
  if (peer_in_anchors.size() != kInferOutNum) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "num of layernorm outs should be 4 in inference, but is [%d] now.",
              peer_in_anchors.size());
      return false;
  }
  if (!UpgradeNodeList(out_anchor, conf_trans_list, matmul_list)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "upgrade matmul_list && conf_trans_list failed!");
    return false;
  }
  // shape_check
  if (!ShapeCheck(ln_node, matmul_list[0], conf_trans_list[0])) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "shape_check failed!");
    return false;
  }
  // dtype check
  auto ln_op_desc = ln_node->GetOpDesc()->GetOutputDesc(0);
  if (ln_op_desc.GetDataType() != ge::DT_FLOAT16) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "ln_node dtype is not fp16, but [%s]!",
        ge::TypeUtils::DataTypeToSerialString(ln_op_desc.GetDataType()).c_str());
    return false;
  }
  // format check
  if (ln_op_desc.GetFormat() != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "ln_node output format is not FRACTAL_NZ, but [%s]!",
        ge::TypeUtils::FormatToSerialString(ln_op_desc.GetFormat()).c_str());
    return false;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AttentionLnQKVONNXFusionPass match success");
  return true;
}

bool AttentionLnQKVONNXFusionPass::ShapeCheck(const ge::NodePtr &ln_node,
                                              const ge::NodePtr &matmul_node,
                                              const ge::NodePtr &conf_trans_node) const {
  vector<int64_t> ln_out_shape = ln_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> matmul_out_shape = matmul_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  // check shape 16 aligned
  bool shape_not_aligned = ln_out_shape[0] % kC0 != 0 || ln_out_shape[1] % kC0 != 0 || matmul_out_shape[1] % kC0 != 0;
  if (shape_not_aligned) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "ln_out_shape (%d, %d) not aligned.", ln_out_shape[0], ln_out_shape[1]);
    return false;
  }
  // check n_shape is supported
  bool unsupported_n_shape = matmul_out_shape[1] != ln_out_shape[1] || (matmul_out_shape[1] != kCandidateN1 &&
      matmul_out_shape[1] != kCandidateN2);
  if (unsupported_n_shape) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "unsupported n_shape [%d] for matmul_qkv.", matmul_out_shape[1]);
    return false;
  }
  if (ln_out_shape[0] % kInferMinMShape1 != 0 && ln_out_shape[0] % kInferMinMShape2 != 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "m_shape should be times of kInferMinMShape [%d]/[%d] in inference, but is [%d] now.",
            kInferMinMShape1, kInferMinMShape2, ln_out_shape[0]);
    return false;
  }
  vector<int64_t> out_shape = conf_trans_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  // seq len should be factor of tiling_m, or the opposite
  bool seq_check = !(kCandidateTilingM1 % out_shape[kMInnerIndex] == 0 ||
                     out_shape[kMInnerIndex] % kCandidateTilingM1 == 0) &&
                    !(kCandidateTilingM2 % out_shape[kMInnerIndex] == 0 ||
                     out_shape[kMInnerIndex] % kCandidateTilingM2 == 0);
  if (out_shape.size() != kOutDimSize || ln_out_shape[0] != out_shape[0] * out_shape[kMInnerIndex] * kC0 ||
      seq_check) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "invalid out_shape (%d, %d)!", out_shape[0], out_shape[kMInnerIndex]);
    return false;
  }
  return true;
}

bool AttentionLnQKVONNXFusionPass::UpgradeNodeList(const ge::OutDataAnchorPtr &out_anchor,
                                                   std::vector<ge::NodePtr> &conf_trans_list,
                                                   std::vector<ge::NodePtr> &matmul_list) {
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  // order for input of LayerNorm: bmm_q/bmm_k/bmm_v/add
  for (int i = 0; i < static_cast<int>(peer_in_anchors.size() - 1); i++) {
    auto next_node = peer_in_anchors.at(i)->GetOwnerNode();
    auto next_matmul_node = next_node;
    if (kOpBatchMatmulV2 != next_matmul_node->GetType()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next node is not matmul, but [%s].", next_matmul_node->GetType().c_str());
      return false;
    }
    auto next_conf_trans_node = next_matmul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    if (kOpConfusionTranspose != next_conf_trans_node->GetType()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next node is not conf_transpose, but [%s].",
          next_conf_trans_node->GetType().c_str());
      return false;
    }
    matmul_list.push_back(next_matmul_node);
    conf_trans_list.push_back(next_conf_trans_node);
  }
  return true;
}

Status AttentionLnQKVONNXFusionPass::CreateAttentionLnQKVNode(ge::ComputeGraph &graph,
                                                              const ge::NodePtr &ln_node,
                                                              const std::vector<ge::NodePtr> &conf_trans_list,
                                                              const std::vector<ge::NodePtr> &matmul_list,
                                                              ge::NodePtr &new_node) {
  auto ln_op_desc = ln_node->GetOpDesc();
  ge::OpDescPtr attention_ln_qkv_desc;
  FUSION_PASS_MAKE_SHARED((attention_ln_qkv_desc = std::make_shared<ge::OpDesc>(
                          ln_node->GetName() + "attention_ln_qkv", kOpAttentionLnQKV)), return FAILED);
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("x", ln_op_desc->GetInputDesc(0).Clone()) != GRAPH_SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to attention_ln_qkv."), return FAILED);
  std::vector<std::string> qkv_names = {"query", "key", "value"};
  std::vector<std::string> matmul_inputs = {"kernel", "bias"};
  // AddInputDesc
  for (unsigned int i = 0; i < matmul_inputs.size(); i++) {
    for (unsigned int j = 0; j < matmul_list.size(); j++) {
      auto pre_out_anchor = matmul_list[j]->GetInDataAnchor(i + 1)->GetPeerOutAnchor();
      auto matmul_input_desc = pre_out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
      string in_desc_name = matmul_inputs[i] + qkv_names[j];
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc(in_desc_name, matmul_input_desc.Clone()) != GRAPH_SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc to attention_ln_qkv."), return FAILED);
    }
    // after adding inputdesc kernel_query/key/value, inputdesc gamma&&beta should be added before inputdesc bias
    if (i == 0) {
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("gamma",
          ln_op_desc->GetInputDesc(kLnMeanIndex).Clone()) != GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
          "failed to add input desc gamma to attention_ln_qkv."), return FAILED);
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("beta",
          ln_op_desc->GetInputDesc(kLnVarIndex).Clone()) != GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
          "failed to add input desc beta to attention_ln_qkv."), return FAILED);
    }
  }
  // Add Outputdesc norm
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("norm", ln_op_desc->GetOutputDesc(0).Clone()) !=
      GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add output desc norm to attention_ln_qkv."),
      return FAILED);
  // Add Outputdesc query/key/value_output
  for (unsigned int i = 0; i < conf_trans_list.size(); i++) {
    auto conf_trans_out_desc = conf_trans_list[i]->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc(qkv_names[i] + "_output", conf_trans_out_desc.Clone()) !=
        GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add output desc to attention_ln_qkv."), return FAILED);
  }
  // output mean&&variance are useful only in training, the outputdesc should be added though
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("mean", ln_op_desc->GetOutputDesc(kLnMeanIndex).Clone()) !=
      GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add output desc mean to attention_ln_qkv."),
      return FAILED);
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("variance",
      ln_op_desc->GetOutputDesc(kLnVarIndex).Clone()) != GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
      "failed to add output desc variance to attention_ln_qkv."), return FAILED);
  new_node = graph.AddNode(attention_ln_qkv_desc);
  FUSION_PASS_CHECK(new_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
      "failed to add attention_ln_qkv to graph."), return FAILED);
  return SUCCESS;
}

Status AttentionLnQKVONNXFusionPass::ProcessLayerNorm(ge::ComputeGraph &graph,
                                                      const ge::NodePtr &ln_node,
                                                      const ge::NodePtr &attention_ln_qkv_node) {
  // inputs of ln are x, gamma, beta, their indexes in new_node are {0, 4, 5}
  vector<unsigned int> ln_input_idx = {kXIndex, kGammaIndex, kBetaIndex};
  for (unsigned int i = 0; i < ln_node->GetAllInDataAnchors().size(); i++) {
    // AddEdge from input_node of ln(x, gamma, beta) to ln
    auto ln_in_data_anchor = ln_node->GetAllInDataAnchors().at(i);
    auto pre_out_anchor = ln_in_data_anchor->GetPeerOutAnchor();
    if (ge::GraphUtils::AddEdge(pre_out_anchor, attention_ln_qkv_node->GetInDataAnchor(ln_input_idx[i])) != SUCCESS) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_ln_qkv node failed.",
          pre_out_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }
  std::vector<ge::NodePtr> remove_node_list = {ln_node};
  // remove edge from ln to add
  auto ln_out_anchor = ln_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(3);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(0), ln_out_anchor) != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from attention_ln_qkv to add
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(0), ln_out_anchor) != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv_node to [%s] failed.",
        ln_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge for output norm success.");
  for (auto &node : remove_node_list) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGE(FUSED_OP_TYPE.c_str(),
        "remove node %s failed.", node->GetName().c_str()), return FAILED);
  }
  return SUCCESS;
}

Status AttentionLnQKVONNXFusionPass::ReplaceAttentionLnQKV(ge::ComputeGraph &graph,
                                                           const ge::NodePtr &ln_node,
                                                           const std::vector<ge::NodePtr> &conf_trans_list,
                                                           const std::vector<ge::NodePtr> &matmul_list,
                                                           ge::NodePtr &attention_ln_qkv_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter ReplaceAttentionLnQKV.");
  // process layer_norm
  if (SUCCESS != ProcessLayerNorm(graph, ln_node, attention_ln_qkv_node)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to process layer_norm.");
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ProcessLayerNorm success.");
  // input starts from x, then kernel_query/key/value
  int index = 1;
  for (unsigned int i = 1; i < kKernelNum; i++) {
    for (unsigned int j = 0; j < matmul_list.size(); j++) {
      auto matmul_node = matmul_list[j];
      auto pre_out_anchor = matmul_node->GetInDataAnchor(i)->GetPeerOutAnchor();
      // AddEdge from input_node of matmul to attention_ln_qkv
      if (ge::GraphUtils::AddEdge(pre_out_anchor, attention_ln_qkv_node->GetInDataAnchor(index++)) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_ln_qkv_node failed.",
            pre_out_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
    // next iteration the index should start from 6
    index = kBiasInputIndex;
  }
  index = 1;
  for (auto trans_node : conf_trans_list) {
    auto trans_out_anchor = trans_node->GetOutDataAnchor(0);
    for (auto peer_in_anchor : trans_out_anchor->GetPeerInDataAnchors()) {
      // AddEdge from attention_ln_qkv to conf_trans_node's output_node
      if (ge::GraphUtils::RemoveEdge(trans_out_anchor, peer_in_anchor) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge from conf_trans_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(index), peer_in_anchor) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
    index++;
  }
  return SUCCESS;
}

REGISTER_PASS("ZAttentionLnQKVONNXFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AttentionLnQKVONNXFusionPass);
} // namespace fe
