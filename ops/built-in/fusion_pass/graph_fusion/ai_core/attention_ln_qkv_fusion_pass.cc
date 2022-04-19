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
static const int KERNEL_NUM = 3;
static const int X_INDEX = 0;
static const int GAMMA_INDEX = 4;
static const int BETA_INDEX = 5;
static const int MEAN_OUT_INDEX = 4;
static const int VAR_OUT_INDEX = 5;
static const int LN_MEAN_INDEX = 1;
static const int LN_VAR_INDEX = 2;
static const int BIAS_INPUT_INDEX = 6;
static const int OUT_DIM_SIZE = 6;
static const int M_INNER_INDEX = 3;
static const int64_t C0 = 16;
static const int64_t MIN_M_SIZE = 8;
static const int64_t BLOCK_NUM_8 = 8;
static const int64_t BLOCK_NUM_32 = 32;
static const string PATTERN_LAYER_NORM = "LayerNorm";
static const string PATTERN_MATMUL = "MatMul";
static const string PATTERN_TRANSDATA = "TransData";
static const string PATTERN_REFORMAT = "ReFormat";
static const string PATTERN_CONFUSIONTRANSPOSE = "ConfusionTransposeD";
static const string LAYERNORM = "LayerNorm";
static const string MATMUL = "MatMulV2";
static const string TRANSDATA = "TransData";
static const string REFORMAT = "ReFormat";
static const string CONFUSIONTRANSPOSE = "ConfusionTransposeD";
static const string AttentionLnQKV = "AttentionLnQKV";
vector<FusionPattern *> AttentionLnQKVFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("AttentionLnQKVFusionPass1");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern1->AddOpDesc(PATTERN_LAYER_NORM, {LAYERNORM})
           .AddOpDesc(PATTERN_TRANSDATA, {TRANSDATA})
           .AddOpDesc(PATTERN_REFORMAT, {REFORMAT})
           .SetInputs(PATTERN_TRANSDATA, {PATTERN_LAYER_NORM})
           .SetInputs(PATTERN_REFORMAT, {PATTERN_TRANSDATA})
           .SetOutput(PATTERN_REFORMAT);
  patterns.push_back(pattern1);
  FusionPattern *pattern2 = new (std::nothrow) FusionPattern("AttentionLnQKVFusionPass2");
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new a pattern object fail."),
    return patterns);
  pattern2->AddOpDesc(PATTERN_LAYER_NORM, {LAYERNORM})
           .AddOpDesc(PATTERN_MATMUL, {MATMUL})
           .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
           .SetInputs(PATTERN_MATMUL, {PATTERN_LAYER_NORM})
           .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_MATMUL})
           .SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern2);
  return patterns;
}

Status AttentionLnQKVFusionPass::Fusion(ge::ComputeGraph &graph,
                                        Mapping &mapping,
                                        vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start AttentionLnQKVFusionPass.");
  ge::NodePtr ln_node = GetNodeFromMapping(PATTERN_LAYER_NORM, mapping);
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
  if (TRANSDATA != trans_node->GetType()) {
    g_trainingFlag = false;
  }
  if (g_trainingFlag) {
    auto reformat_node = trans_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    if (REFORMAT != reformat_node->GetType()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "second node [%s] match failed.", reformat_node->GetName().c_str());
      return false;
    }
    out_anchor = reformat_node->GetOutDataAnchor(0);
  }
  if (!UpgradeNodeList(out_anchor, conf_trans_list, matmul_list)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Upgade matmul_list && conf_trans_list failed!");
    return false;
  }
  // shape_check
  vector<int64_t> ln_out_shape = ln_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  if (ln_out_shape[0] % C0 != 0 || ln_out_shape[1] % C0 != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ln_out_shape not aligned.");
    return false;
  }
  if (ln_out_shape[0] % (C0 * MIN_M_SIZE * BLOCK_NUM_8) != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "C0 * MIN_M_SIZE * BLOCK_NUM_8 should be factor of m_shape.");
    return false;
  }
  if (g_trainingFlag and ln_out_shape[0] % (C0 * MIN_M_SIZE * BLOCK_NUM_32) != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "in training, C0 * MIN_M_SIZE * BLOCK_NUM_32 should be factor of m_shape.");
    return false;
  }
  vector<int64_t> out_shape = conf_trans_list[0]->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  if (out_shape.size() != OUT_DIM_SIZE || ln_out_shape[0] != out_shape[0] * out_shape[M_INNER_INDEX] * C0 ||
      out_shape[M_INNER_INDEX] % MIN_M_SIZE != 0) {
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
  if (peer_in_anchors.size() <= KERNEL_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "in training, output nodes nums of reformat unmatched!");
    return false;
  }
  // in inference, out index of matmul starts at 1
  for (int i = !g_trainingFlag; i < KERNEL_NUM + !g_trainingFlag; i++) {
    auto next_node = peer_in_anchors.at(i)->GetOwnerNode();
    auto next_matmul_node = next_node;
    // in training, pre node of matmul is trans_data
    if (g_trainingFlag) {
      if (TRANSDATA != next_node->GetType()) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "next node of ReFormat is not TransData, but [%s].",
            next_node->GetType().c_str());
        return false;
      }
      next_matmul_node = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    }
    if (MATMUL != next_matmul_node->GetType()) {
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
    if (CONFUSIONTRANSPOSE != next_conf_trans_node->GetType()) {
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
                          ln_node->GetName() + "attention_ln_qkv", AttentionLnQKV)), return FAILED);
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
          ln_op_desc->GetInputDesc(LN_MEAN_INDEX).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
          "failed to add input desc gamma to attention_ln_qkv."), return FAILED);
      FUSION_PASS_CHECK(attention_ln_qkv_desc->AddInputDesc("beta",
          ln_op_desc->GetInputDesc(LN_VAR_INDEX).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
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
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("mean", ln_op_desc->GetOutputDesc(LN_MEAN_INDEX).Clone()) !=
      GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "failed to add output desc mean to attention_ln_qkv."),
      return FAILED);
  FUSION_PASS_CHECK(attention_ln_qkv_desc->AddOutputDesc("variance",
      ln_op_desc->GetOutputDesc(LN_VAR_INDEX).Clone()) != GRAPH_SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "failed to add output desc variance to attention_ln_qkv."), return FAILED);
  new_node = graph.AddNode(attention_ln_qkv_desc);
  FUSION_PASS_CHECK(new_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
      "failed to add attention_ln_qkv to graph."), return FAILED);
  return SUCCESS;
}

Status AttentionLnQKVFusionPass::ProcessLayerNormBackprop(const ge::NodePtr &ln_node,
                                                          const ge::NodePtr &attention_ln_qkv_node,
                                                          std::vector<ge::NodePtr> &remove_node_list) {
  auto ln_grad_in_anchor = ln_node->GetOutDataAnchor(LN_MEAN_INDEX)->GetPeerInDataAnchors().at(0);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(LN_MEAN_INDEX), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from ln to output mean
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(MEAN_OUT_INDEX), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge from attention_ln_qkv to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  ln_grad_in_anchor = ln_node->GetOutDataAnchor(LN_VAR_INDEX)->GetPeerInDataAnchors().at(0);
  if (ge::GraphUtils::RemoveEdge(ln_node->GetOutDataAnchor(LN_VAR_INDEX), ln_grad_in_anchor) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge from ln_node to [%s] failed.",
        ln_grad_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  // AddEdge from ln to output variance
  if (ge::GraphUtils::AddEdge(attention_ln_qkv_node->GetOutDataAnchor(VAR_OUT_INDEX), ln_grad_in_anchor) != SUCCESS) {
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
  vector<unsigned int> ln_input_idx = {X_INDEX, GAMMA_INDEX, BETA_INDEX};
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
  for (unsigned int i = 1; i < KERNEL_NUM; i++) {
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
    index = BIAS_INPUT_INDEX;
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
