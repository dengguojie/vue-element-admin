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
 * \file attention_ln_qkv_fusion_pass.cc
 * \brief the pass will turn three conjuction matmul_confusionTranspose into a attention_ln_qkv
 *  *  * Training pattern:
 *                        LayerNorm
 *                      /     |     \
 *                   var  TransData  mean
 *                            |
 *                         ReFormat
 *                      /     |     \
 *              TransData TransData TransData  =>   attention_ln_qkv
 *                  |         |          |
 *                mm_q      mm_k       mm_v
 *                  |         |          |
 *          ConTrans_q   ConTrans_k   ConTrans_v
 *
 *  *  *  *Inference pattern:
 *                        LayerNorm
 *                      /     |     \
 *                    var     |     mean
 *                         /  |  \            =>   attention_ln_qkv
 *                        /   |   \
 *                     mm_q  mm_k  mm_v
 *                     /      |     \
 *           ConTrans_q  ConTrans_k  ConTrans_v
 */
#include "attention_ln_qkv_fusion_pass.h"
#include "anchor_util.h"
#include "graph/utils/graph_utils.h"
#include "common/util/platform_info.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static bool g_trainingFlag = true;
static const int kKernelNum = 3;
static const int kXIndex = 0;
static const int kGammaIndex = 4;
static const int kBetaIndex = 5;
static const int kMeanOutIndex = 4;
static const int kVarOutIndex = 5;
static const int kLnMeanIndex = 1;
static const int kLnVarIndex = 2;
static const int kBiasInputIndex = 6;
static const int kOutDimSize = 6;
static const int kMInnerIndex = 3;
static const int64_t kC0 = 16;
static const int64_t kInferMinMShape = 12288;
static const int64_t kCandidateN1 = 1024;
static const int64_t kCandidateN2 = 768;
static const int64_t kCandidateTilingM1 = 12;
static const int64_t kCandidateTilingM2 = 8;
static const string kPatternLayernorm = "LayerNorm";
static const string kPatternMatmul = "MatMul";
static const string kPatternTransdata = "TransData";
static const string kPatternReformat = "ReFormat";
static const string kPatternConfusionTranspose = "ConfusionTransposeD";
static const string kOpLayernorm = "LayerNorm";
static const string kOpMatmul = "MatMulV2";
static const string kOpTransdata = "TransData";
static const string kOpReformat = "ReFormat";
static const string kOpConfusionTranspose = "ConfusionTransposeD";
static const string kOpAttentionLnQKV = "AttentionLnQKV";
vector<FusionPattern *> AttentionLnQKVFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("AttentionLnQKVFusionPass1");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern1->AddOpDesc(kPatternLayernorm, {kOpLayernorm})
           .AddOpDesc(kPatternTransdata, {kOpTransdata})
           .AddOpDesc(kPatternReformat, {kOpReformat})
           .SetInputs(kPatternTransdata, {kPatternLayernorm})
           .SetInputs(kPatternReformat, {kPatternTransdata})
           .SetOutput(kPatternReformat);
  patterns.push_back(pattern1);
  FusionPattern *pattern2 = new (std::nothrow) FusionPattern("AttentionLnQKVFusionPass2");
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern2->AddOpDesc(kPatternLayernorm, {kOpLayernorm})
           .AddOpDesc(kPatternMatmul, {kOpMatmul})
           .AddOpDesc(kPatternConfusionTranspose, {kOpConfusionTranspose})
           .SetInputs(kPatternMatmul, {kPatternLayernorm})
           .SetInputs(kPatternConfusionTranspose, {kPatternMatmul})
           .SetOutput(kPatternConfusionTranspose);
  patterns.push_back(pattern2);
  return patterns;
}

Status AttentionLnQKVFusionPass::Fusion(ge::ComputeGraph &graph,
                                        Mapping &mapping,
                                        vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start AttentionLnQKVFusionPass.");
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
  if (optional_info.soc_version != "Ascend710") {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "platform not supported.");
    return NOT_CHANGED;
  }

  std::vector<ge::NodePtr> conf_trans_list;
  std::vector<ge::NodePtr> matmul_list;
  if (!IsMatch(ln_node, conf_trans_list, matmul_list)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Match AttentionLnQKVFusionPass failed.");
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
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGW(FUSED_OP_TYPE.c_str(),
                      "remove [%s] node failed.", node->GetName().c_str()), return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End AttentionLnQKVFusionPass.");
  fusion_nodes.push_back(attention_ln_qkv_node);
  return SUCCESS;
}

bool AttentionLnQKVFusionPass::IsMatch(const ge::NodePtr &ln_node,
                                       std::vector<ge::NodePtr> &conf_trans_list,
                                       std::vector<ge::NodePtr> &matmul_list) {
  // pattern check
  auto out_anchor = ln_node->GetOutDataAnchor(0);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  // training/inference both has 2 outputs
  if (peer_in_anchors.size() <= 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "output nodes nums of LN unmatched!");
    return false;
  }
  // training pattern is:
  //   layer_norm-> add(1) + transdata-> reformat-> transdata->mm_q + transdata->mm_k + transdata->mm_v
  auto trans_node = peer_in_anchors.at(0)->GetOwnerNode();
  if (kOpTransdata != trans_node->GetType()) {
    g_trainingFlag = false;
  }
  if (g_trainingFlag) {
    auto reformat_node = trans_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    if (kOpReformat != reformat_node->GetType()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "second node [%s] match failed.", reformat_node->GetName().c_str());
      return false;
    }
    out_anchor = reformat_node->GetOutDataAnchor(0);
  }
  if (!UpgradeNodeList(out_anchor, conf_trans_list, matmul_list)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "upgrade matmul_list && conf_trans_list failed!");
    return false;
  }
  // shape_check
  vector<int64_t> ln_out_shape = ln_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> matmul_out_shape = matmul_list[0]->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  // check shape 16 aligned
  bool shape_not_aligned = ln_out_shape[0] % kC0 != 0 || ln_out_shape[1] % kC0 != 0 || matmul_out_shape[1] % kC0 != 0;
  if (shape_not_aligned) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ln_out_shape not aligned.");
    return false;
  }
  // check n_shape is supported
  bool unsupported_n_shape = matmul_out_shape[1] != ln_out_shape[1] || (matmul_out_shape[1] != kCandidateN1 &&
      matmul_out_shape[1] != kCandidateN2);
  if (unsupported_n_shape) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "unsupported n_shape for matmul_qkv.");
    return false;
  }
  if (ln_out_shape[0] % kInferMinMShape != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "in inference, m_shape should be times of 12288.");
    return false;
  }
  vector<int64_t> out_shape = conf_trans_list[0]->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  // seq len should be factor of tiling_m, or the opposite
  bool seq_check = !(kCandidateTilingM1 % out_shape[kMInnerIndex] == 0 ||
                     out_shape[kMInnerIndex] % kCandidateTilingM1 == 0) &&
                    !(kCandidateTilingM2 % out_shape[kMInnerIndex] == 0 ||
                     out_shape[kMInnerIndex] % kCandidateTilingM2 == 0);
  if (out_shape.size() != kOutDimSize || ln_out_shape[0] != out_shape[0] * out_shape[kMInnerIndex] * kC0 ||
      seq_check) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "invalid out_shape!");
    return false;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AttentionLnQKVFusionPass match success");
  return true;
}

bool AttentionLnQKVFusionPass::UpgradeNodeList(const ge::OutDataAnchorPtr &out_anchor,
                                               std::vector<ge::NodePtr> &conf_trans_list,
                                               std::vector<ge::NodePtr> &matmul_list) {
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "output size of ln_node is [%d].", peer_in_anchors.size());
  if (peer_in_anchors.size() <= kKernelNum) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "in training, output nodes nums of reformat unmatched!");
    return false;
  }
  // in inference, out index of matmul starts at 1
  for (int i = !g_trainingFlag; i < kKernelNum + !g_trainingFlag; i++) {
    auto next_node = peer_in_anchors.at(i)->GetOwnerNode();
    auto next_matmul_node = next_node;
    // in training, pre node of matmul is trans_data
    if (g_trainingFlag) {
      if (kOpTransdata != next_node->GetType()) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "next node of ReFormat is not TransData, but [%s].",
            next_node->GetType().c_str());
        return false;
      }
      next_matmul_node = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    }
    if (kOpMatmul != next_matmul_node->GetType()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "next node is not matmul, but [%s].", next_matmul_node->GetType().c_str());
      return false;
    }
    bool trans_a = false;
    bool trans_b = false;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(next_matmul_node->GetOpDesc(), "transpose_x1", trans_a),
        OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to get attr trans_a."), return false);
    FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(next_matmul_node->GetOpDesc(), "transpose_x2", trans_b),
        OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to get attr trans_b."), return false);
    if (trans_a || trans_b) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "trans_a/tran_b of matmul_node matches failed.");
      return false;
    }
    auto next_conf_trans_node = next_matmul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    if (kOpConfusionTranspose != next_conf_trans_node->GetType()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "next node is not conf_transpose, but [%s].",
          next_conf_trans_node->GetType().c_str());
      return false;
    }
    matmul_list.push_back(next_matmul_node);
    conf_trans_list.push_back(next_conf_trans_node);
  }
  return true;
}

Status AttentionLnQKVFusionPass::CreateAttentionLnQKVNode(ge::ComputeGraph &graph,
                                                          const ge::NodePtr &ln_node,
                                                          const std::vector<ge::NodePtr> &conf_trans_list,
                                                          const std::vector<ge::NodePtr> &matmul_list,
                                                          ge::NodePtr &new_node) {
  auto ln_op_desc = ln_node->GetOpDesc();
  ge::OpDescPtr attention_ln_qkv_desc;
  FUSION_PASS_MAKE_SHARED((attention_ln_qkv_desc = std::make_shared<ge::OpDesc>(
                          ln_node->GetName() + "attention_ln_qkv", kOpAttentionLnQKV)), return FAILED);
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("x", ln_op_desc->GetInputDesc(0).Clone()) != GRAPH_SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add input desc x to attention_ln_qkv."), return FAILED);
  std::vector<std::string> qkv_names = {"query", "key", "value"};
  std::vector<std::string> matmul_inputs = {"kernel", "bias"};
  // AddInputDesc
  for (unsigned int i = 0; i < matmul_inputs.size(); i++) {
    for (unsigned int j = 0; j < matmul_list.size(); j++) {
      auto pre_out_anchor = matmul_list[j]->GetInDataAnchor(i + 1)->GetPeerOutAnchor();
      auto matmul_input_desc = pre_out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
      string in_desc_name = matmul_inputs[i] + qkv_names[j];
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc(in_desc_name, matmul_input_desc.Clone()) != GRAPH_SUCCESS,
          OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add input desc to attention_ln_qkv."), return FAILED);
    }
    // after adding inputdesc kernel_query/key/value, inputdesc gamma&&beta should be added before inputdesc bias
    if (i == 0) {
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("gamma",
          ln_op_desc->GetInputDesc(kLnMeanIndex).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
          "failed to add input desc gamma to attention_ln_qkv."), return FAILED);
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("beta",
          ln_op_desc->GetInputDesc(kLnVarIndex).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
          "failed to add input desc beta to attention_ln_qkv."), return FAILED);
    }
  }
  // Add Outputdesc norm
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("norm", ln_op_desc->GetOutputDesc(0).Clone()) !=
      GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add output desc norm to attention_ln_qkv."),
      return FAILED);
  // Add Outputdesc query/key/value_output
  for (unsigned int i = 0; i < conf_trans_list.size(); i++) {
    auto conf_trans_out_desc = conf_trans_list[i]->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc(qkv_names[i] + "_output", conf_trans_out_desc.Clone()) !=
        GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add output desc to attention_ln_qkv."), return FAILED);
  }
  // output mean&&variance are useful only in training, the outputdesc should be added though
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("mean", ln_op_desc->GetOutputDesc(kLnMeanIndex).Clone()) !=
      GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add output desc mean to attention_ln_qkv."),
      return FAILED);
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("variance",
      ln_op_desc->GetOutputDesc(kLnVarIndex).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "failed to add output desc variance to attention_ln_qkv."), return FAILED);
  new_node = graph.AddNode(attention_ln_qkv_desc);
  FUSION_PASS_CHECK(new_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "failed to add attention_ln_qkv to graph."), return FAILED);
  return SUCCESS;
}

Status AttentionLnQKVFusionPass::ProcessLayerNormBackprop(const ge::NodePtr &ln_node,
                                                          const ge::NodePtr &attention_ln_qkv_node,
                                                          std::vector<ge::NodePtr> &remove_node_list) {
  auto ln_grad_in_anchor = ln_node->GetOutDataAnchor(kLnMeanIndex)->GetPeerInDataAnchors().at(0);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(kLnMeanIndex), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from ln to output mean
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(kMeanOutIndex), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  ln_grad_in_anchor = ln_node->GetOutDataAnchor(kLnVarIndex)->GetPeerInDataAnchors().at(0);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(kLnVarIndex), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from ln to output variance
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(kVarOutIndex), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv_node to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge for mean/variance success.");
  // Add trans_data and reformat to remove_node_list
  auto trans_node = ln_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  auto reformat_node = trans_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  remove_node_list.push_back(trans_node);
  remove_node_list.push_back(reformat_node);
  // Add trans_data node after reformat node to remove_node_list
  for (unsigned int i = 0; i < reformat_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(); i++) {
    auto trans_data_node = reformat_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(i)->GetOwnerNode();
    remove_node_list.push_back(trans_data_node);
  }
  return SUCCESS;
}

Status AttentionLnQKVFusionPass::ProcessLayerNorm(ge::ComputeGraph &graph,
                                                  const ge::NodePtr &ln_node,
                                                  const ge::NodePtr &attention_ln_qkv_node) {
  // inputs of ln are x, gamma, beta, their indexes in new_node are {0, 4, 5}
  vector<unsigned int> ln_input_idx = {kXIndex, kGammaIndex, kBetaIndex};
  for (unsigned int i = 0; i < ln_node->GetAllInDataAnchors().size(); i++) {
    // AddEdge from input_node of ln(x, gamma, beta) to ln
    auto ln_in_data_anchor = ln_node->GetAllInDataAnchors().at(i);
    auto pre_out_anchor = ln_in_data_anchor->GetPeerOutAnchor();
    if (ge::GraphUtils::AddEdge(pre_out_anchor, attention_ln_qkv_node->GetInDataAnchor(ln_input_idx[i])) != SUCCESS) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_ln_qkv node failed.",
          pre_out_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }
  std::vector<ge::NodePtr> remove_node_list= {ln_node};
  // in training, mean&&variance should be passed to LayerNormBackprop
  if (g_trainingFlag && SUCCESS != ProcessLayerNormBackprop(ln_node, attention_ln_qkv_node, remove_node_list)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ProcessLayerNormBackprop failed in training.");
    return FAILED;
  }
  // remove edge from ln to add
  auto ln_out_anchor = ln_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(g_trainingFlag);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(0), ln_out_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from attention_ln_qkv to add
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(0), ln_out_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv_node to [%s] failed.",
        ln_out_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge for output norm success.");
  for (auto &node : remove_node_list) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node), OP_LOGW(FUSED_OP_TYPE.c_str(),
        "remove node %s failed.", node->GetName().c_str()), return FAILED);
  }
  return SUCCESS;
}

Status AttentionLnQKVFusionPass::ReplaceAttentionLnQKV(ge::ComputeGraph &graph,
                                                       const ge::NodePtr &ln_node,
                                                       const std::vector<ge::NodePtr> &conf_trans_list,
                                                       const std::vector<ge::NodePtr> &matmul_list,
                                                       ge::NodePtr &attention_ln_qkv_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter ReplaceAttentionLnQKV.");
  // process layer_norm
  if (SUCCESS != ProcessLayerNorm(graph, ln_node, attention_ln_qkv_node)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to process layer_norm.");
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
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from [%s] to attention_ln_qkv_node failed.",
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
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from conf_trans_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(index), peer_in_anchor) != SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv_node to [%s] failed.",
            peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
    index++;
  }
  return SUCCESS;
}

REGISTER_PASS("ZAttentionLnQKVFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AttentionLnQKVFusionPass);
} // namespace fe
