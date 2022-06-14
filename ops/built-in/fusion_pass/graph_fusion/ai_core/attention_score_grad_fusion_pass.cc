/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file attention_score_grad_fusion_pass.cc
 * pattern:
 *    x        const0                           const0
 *     \      /                                 /
 *   batch_matmul    const1          x       pad0      const1
 *              \    /                \      /         /
 *               add                 batch_matmul    pad1
 *                |                             \    /
 *               relu    const2  ->              add
 *                 \    /                         |
 *               reduce_mean                     relu    const2
 *                                                 \    /
 *                                               reduce_mean
 *                                                   |
 *                                                 slice
 */
#include <stdlib.h>
#include "attention_score_grad_fusion_pass.h"

#include "anchor_util.h"
#include "error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "external/graph/operator_factory.h"

namespace fe {
namespace {
static const char PATTERN_INPUTCONFUSIONTRANSPOSE[] = "input_confusion_transpose";
static const char PATTERN_BATCHMATMUL[] = "batch_matmul";
static const char PATTERN_BATCHMATMUL2[] = "batch_matmul2";
static const char PATTERN_DROPOUTDOMASKV3[] = "drop_out_do_mask_v3";
static const char PATTERN_SOFTMAXGRADEXT[] = "softmax_grad_ext";
static const char PATTERN_CONFUSIONTRANSPOSE1[] = "confusion_transpose1";
static const char PATTERN_CONFUSIONTRANSPOSE2[] = "confusion_transpose2";
static const char PATTERN_CONFUSIONTRANSPOSE3[] = "confusion_transpose3";


static const char BATCHMATMULV2[] = "BatchMatMulV2";
static const char DROPOUTDOMASKV3D[] = "DropOutDoMaskV3D";
static const char SOFTMAXGRADEXT[] = "SoftmaxGradExt";
static const char CONFUSIONTRANSPOSE[] = "ConfusionTransposeD";

static const int ALIGN_UNIT = 16;
static const int CONFUSION_DIM_ONE = 12288;
static const int CONFUSION_DIM_TWO = 768;
static const int NUM_TWO = 2;
static const int NUM_THREE = 3;
static const int NUM_FOUR = 4;
static const int NUM_FIVE = 5;
static const int NUM_SIX = 6;

static const char kNameFusionPass[] = "ZAttentionScoreGradFusionPass";
}  // namespace

vector<FusionPattern *> ZAttentionScoreGradFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern.");
  pattern->AddOpDesc(PATTERN_INPUTCONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_DROPOUTDOMASKV3, {DROPOUTDOMASKV3D})
    .AddOpDesc(PATTERN_SOFTMAXGRADEXT, {SOFTMAXGRADEXT})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE1, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_BATCHMATMUL, {PATTERN_INPUTCONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_DROPOUTDOMASKV3, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXGRADEXT, {PATTERN_DROPOUTDOMASKV3})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXGRADEXT})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE1, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE1);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  return patterns;
}

Status ZAttentionScoreGradFusionPass::AddInputDescForBsb(std::shared_ptr<ge::OpDesc> bsb_desc) {
  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("attention_score",
                                           *(batch_matmul_node4->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input0 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("dx",
                                           *(batch_matmul_node4->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input1 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("query",
                                           *(batch_matmul_node1->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input2 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("key",
                                           *(batch_matmul_node2->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input3 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("value",
                                           *(batch_matmul_node3->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input4 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("scale",
                                           *(softmax_grad_ext_node->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("drop_mask",
                                           *(drop_out_do_mask_v3_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input6 of AttentionScore failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddOutputDescForBsb(std::shared_ptr<ge::OpDesc> bsb_desc) {
  FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("value_dw",
                                            *(confusion_transpose_node2->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("query_dx",
                                            *(confusion_transpose_node->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("key_dw",
                                            *(confusion_transpose_node1->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::DeleteOldNode(ge::ComputeGraph &graph) {
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node3),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node3->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node4),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node4->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusion_transpose_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusion_transpose_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusion_transpose_node1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusion_transpose_node1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusion_transpose_node2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusion_transpose_node2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(softmax_grad_ext_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     softmax_grad_ext_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(drop_out_do_mask_v3_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     drop_out_do_mask_v3_node->GetName().c_str()),
      return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddControlEdgesForBsbNode(ge::NodePtr bsb_node) {
  // add control edge
  if (softmax_grad_ext_node->GetInControlAnchor()) {
    for (auto out_control_anchor : softmax_grad_ext_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      if (drop_out_do_mask_v3_node->GetName().c_str() == out_control_anchor->GetOwnerNode()->GetName().c_str()) {
        continue;
      } else {
        FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, bsb_node->GetInControlAnchor()) != SUCCESS,
                            CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
                            return FAILED);
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_control_anchor,
                                                       softmax_grad_ext_node->GetInControlAnchor()) != SUCCESS,
                            CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                                           "Remove softmax_node node input control edge failed."),
                            return FAILED);
      }
    }
  }
  if (softmax_grad_ext_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : softmax_grad_ext_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmax_grad_ext_node->GetOutControlAnchor(),
                                                   in_control_anchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove softmax node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsb_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }

  if (drop_out_do_mask_v3_node->GetInControlAnchor()) {
    for (auto out_control_anchor : drop_out_do_mask_v3_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, bsb_node->GetInControlAnchor()) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_control_anchor,
                                                   drop_out_do_mask_v3_node->GetInControlAnchor()) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                                       "Remove drop_out node input control edge failed."),
                        return FAILED);
    }
  }
  if (drop_out_do_mask_v3_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : drop_out_do_mask_v3_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(drop_out_do_mask_v3_node->GetOutControlAnchor(),
                                                   in_control_anchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove drop_out node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsb_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddEdgesForBsbNode(ge::NodePtr bsb_node) {
 // add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmax_grad_ext_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node4->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node1->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_TWO)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node2->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_THREE)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node3->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_FOUR)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmax_grad_ext_node->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(drop_out_do_mask_v3_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_SIX)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);

  AddOutputEdgeForNode(confusion_transpose_node2, bsb_node, 0, 0);

  AddOutputEdgeForNode(confusion_transpose_node, bsb_node, 0, 1);

  AddOutputEdgeForNode(confusion_transpose_node1, bsb_node, 0, NUM_TWO);

  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::SetAttrsForBsb(std::shared_ptr<ge::OpDesc> bsb_desc) {
  float keep_prob = 0;
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetFloat(drop_out_do_mask_v3_node->GetOpDesc(), "keep_prob", keep_prob),
    OP_LOGE(kNameFusionPass, "Failed to get keep_prod of %s.", drop_out_do_mask_v3_node->GetName().c_str()),
    return FAILED);

  ge::AttrUtils::SetFloat(bsb_desc, "keep_prob", keep_prob);

  return SUCCESS;
}

bool ZAttentionScoreGradFusionPass::CheckNodeShapeSupported() {
  return (CheckSoftmaxGradShapeSupported() || CheckBatchMatMulShapeSupported(batch_matmul_node1) ||
          CheckBatchMatMulShapeSupported(batch_matmul_node2) || CheckBatchMatMulShapeSupported(batch_matmul_node3) ||
          CheckSpcBatchMatMulShapeSupported(batch_matmul_node4));
}

bool ZAttentionScoreGradFusionPass::CheckSpcBatchMatMulShapeSupported(const ge::NodePtr bmm_node) {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", bmm_node->GetName().c_str());
  vector<int64_t> bmm_input1_dims = bmm_node->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmm_input1_dims.empty(),
                    OP_LOGI(kNameFusionPass, "Get bmm input1 shape not success."),
                    return false);
  vector<int64_t> bmm_input0_dims = bmm_node->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmm_input0_dims.empty(),
                    OP_LOGI(kNameFusionPass, "Get bmm input1 shape not success."),
                    return false);
  vector<int64_t> bmm1_dims = batch_matmul_node1->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> softmax_shape_dims = softmax_grad_ext_node->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  if ((bmm_input1_dims != bmm1_dims) || (bmm_input0_dims != softmax_shape_dims)) {
    return true;
  }
  return false;
}

bool ZAttentionScoreGradFusionPass::CheckBatchMatMulShapeSupported(const ge::NodePtr bmm_node) {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", bmm_node->GetName().c_str());
  vector<int64_t> bmm_dims = bmm_node->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  FUSION_PASS_CHECK(bmm_dims.empty(),
                    OP_LOGI(kNameFusionPass, "Get bmm shape not success."),
                    return false);
  int64_t bmm_dim2 = bmm_dims[NUM_TWO];
  int64_t bmm_dim3 = bmm_dims[NUM_THREE];
  int64_t batch_dim3_limit = 64;
  bool shape_not_matched = false;
  int64_t dim_size = bmm_dims.size();
  auto bmm_format = bmm_node->GetOpDesc()->GetInputDesc(1).GetFormat();
  if (dim_size == NUM_FOUR && bmm_dim3 == batch_dim3_limit) {
    shape_not_matched = false;
  } else if ((dim_size == NUM_SIX) && (bmm_dim2 == NUM_FOUR) && (bmm_format == ge::FORMAT_FRACTAL_NZ)) {
    shape_not_matched = false;
  } else {
    shape_not_matched = true;
  }

  return shape_not_matched;
}

bool ZAttentionScoreGradFusionPass::CheckSoftmaxGradShapeSupported() {
  OP_LOGD(kNameFusionPass, "Start check shape of %s.", softmax_grad_ext_node->GetName().c_str());
  vector<int64_t> softmax_shape_dims = softmax_grad_ext_node->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(softmax_shape_dims.empty(),
                    OP_LOGI(kNameFusionPass, "Get softmax_dims not success."),
                    return NOT_CHANGED);
  int64_t batch_dim0 = softmax_shape_dims[0];
  int64_t batch_dim1 = softmax_shape_dims[1];
  int64_t batch_dim2 = softmax_shape_dims[NUM_TWO];
  int64_t batch_dim3 = softmax_shape_dims[NUM_THREE];
  int64_t dim_size = softmax_shape_dims.size();
  int64_t seq_dim_limit_factor = 32;
  int64_t seq_dim_limit_factor_nz = 2;
  bool shape_not_matched = false;

  if (dim_size != NUM_FOUR && dim_size != NUM_SIX) {
    OP_LOGI(kNameFusionPass, "Shape not match, graph not change.");
    shape_not_matched = true;
  }

  if (batch_dim1 != ALIGN_UNIT || batch_dim2 != batch_dim3) {
    shape_not_matched = true;
  }
  auto softmax_format = softmax_grad_ext_node->GetOpDesc()->GetInputDesc(0).GetFormat();
  if (dim_size == NUM_FOUR) {
    if (batch_dim2 % seq_dim_limit_factor != 0) {
      shape_not_matched = true;
    }
    if (batch_dim0 * batch_dim2 != CONFUSION_DIM_ONE) {
      shape_not_matched = true;
    }
  } else if (dim_size == NUM_SIX && softmax_format == ge::FORMAT_FRACTAL_NZ) {
    if (batch_dim2 % seq_dim_limit_factor_nz != 0) {
      shape_not_matched = true;
    }
    if (batch_dim0 * batch_dim3 != CONFUSION_DIM_TWO) {
      shape_not_matched = true;
    }
  } else {
    shape_not_matched = true;
  }

  return shape_not_matched;
}

Status ZAttentionScoreGradFusionPass::CreateAttentionDesc() {
  FUSION_PASS_MAKE_SHARED(
    bsb_desc = std::make_shared<ge::OpDesc>(softmax_grad_ext_node->GetName() + "/AttentionScoreGrad",
                                            "AttentionScoreGrad"),
    return FAILED);

  FUSION_PASS_CHECK(AddInputDescForBsb(bsb_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input desc for bsb failed."),
                    return FAILED);
  FUSION_PASS_CHECK(AddOutputDescForBsb(bsb_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output desc for bsb failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SetAttrsForBsb(bsb_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Set Attr for bsb failed."),
                    return FAILED);
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::GetFusionNode(Mapping &mapping, ge::NodePtr input_confusion_transpose) {
  batch_matmul_node1 = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(batch_matmul_node1 == nullptr,
                    OP_LOGI(kNameFusionPass, "The graph has been fusion."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(batch_matmul_node1->GetInNodes().size() == 0,
                    OP_LOGI(kNameFusionPass, "Get batch_matmul_node1 not success."),
                    return NOT_CHANGED);

  for (auto netout_in_data_anchor : input_confusion_transpose->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr temp_batch_node = netout_in_data_anchor->GetOwnerNode();
    if (temp_batch_node->GetName().c_str() != batch_matmul_node1->GetName().c_str()) {
      batch_matmul_node4 = temp_batch_node;
      break;
    }
  }
  FUSION_PASS_CHECK(batch_matmul_node4 == nullptr,
                    OP_LOGI(kNameFusionPass, "The batch_matmul_node4 is nullptr."),
                    return NOT_CHANGED);
  for (auto netout_in_data_anchor : batch_matmul_node4->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr temp_batch_node = netout_in_data_anchor->GetOwnerNode();
    confusion_transpose_node2 = temp_batch_node;
  }
  FUSION_PASS_CHECK(confusion_transpose_node2 == nullptr,
                    OP_LOGI(kNameFusionPass, "The confusion_transpose_node2 is nullptr."),
                    return NOT_CHANGED);
  drop_out_do_mask_v3_node = GetNodeFromMapping(PATTERN_DROPOUTDOMASKV3, mapping);
  FUSION_PASS_CHECK(drop_out_do_mask_v3_node == nullptr,
                    OP_LOGI(kNameFusionPass, "Get drop_out_do_mask_v3_node not success."),
                    return NOT_CHANGED);

  softmax_grad_ext_node = GetNodeFromMapping(PATTERN_SOFTMAXGRADEXT, mapping);
  FUSION_PASS_CHECK(softmax_grad_ext_node == nullptr,
                    OP_LOGI(kNameFusionPass, "Get softmax_grad_ext_node not success."),
                    return NOT_CHANGED);

  batch_matmul_node2 = GetNodeFromMapping(PATTERN_BATCHMATMUL2, mapping);
  FUSION_PASS_CHECK(batch_matmul_node2 == nullptr,
                    OP_LOGI(kNameFusionPass, "Get batch_matmul_node2 not success."),
                    return NOT_CHANGED);

  confusion_transpose_node = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE1, mapping);
  FUSION_PASS_CHECK(confusion_transpose_node == nullptr,
                    OP_LOGI(kNameFusionPass, "Get confusion_transpose_node not success."),
                    return NOT_CHANGED);

  for (auto netout_in_data_anchor : softmax_grad_ext_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr temp_batch_node = netout_in_data_anchor->GetOwnerNode();
    if (temp_batch_node->GetName().c_str() != batch_matmul_node2->GetName().c_str()) {
      batch_matmul_node3 = temp_batch_node;
      break;
    }
  }
  FUSION_PASS_CHECK(batch_matmul_node3 == nullptr,
                    OP_LOGI(kNameFusionPass, "The batch_matmul_node3 is nullptr."),
                    return NOT_CHANGED);
  bool bmm2_transpose_a = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(batch_matmul_node2->GetOpDesc(), "adj_x1", bmm2_transpose_a),
                    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batch_matmul_node2->GetName().c_str()),
                    return NOT_CHANGED);
  for (auto netout_in_data_anchor : batch_matmul_node3->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr temp_batch_node = netout_in_data_anchor->GetOwnerNode();
    confusion_transpose_node1 = temp_batch_node;
  }
  FUSION_PASS_CHECK(confusion_transpose_node1 == nullptr,
                    OP_LOGI(kNameFusionPass, "The confusion_transpose_node1 is nullptr."),
                    return NOT_CHANGED);
  if (bmm2_transpose_a) {
    ge::NodePtr temp_bmm_node = batch_matmul_node2;
    batch_matmul_node2 = batch_matmul_node3;
    batch_matmul_node3 = temp_bmm_node;

    temp_bmm_node = confusion_transpose_node1;
    confusion_transpose_node1 = confusion_transpose_node;
    confusion_transpose_node = temp_bmm_node;
  }
  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGI(kNameFusionPass, "Start ZAttentionScoreGradFusionPass.");
  ge::NodePtr input_confusion_transpose = GetNodeFromMapping(PATTERN_INPUTCONFUSIONTRANSPOSE, mapping);
  FUSION_PASS_CHECK(input_confusion_transpose == nullptr,
                    OP_LOGI(kNameFusionPass, "Get input_confusion_transpose not success."),
                    return NOT_CHANGED);

  /*
  * Get fused node in mapping assign global var.
  */
  OP_LOGD(kNameFusionPass, "Start GetFusionNode.");
  FUSION_PASS_CHECK(SUCCESS != GetFusionNode(mapping, input_confusion_transpose),
                    OP_LOGI(kNameFusionPass, "Get fused node not success."),
                    return NOT_CHANGED);
  /*
  * Check node Shape is supported.
  */
  OP_LOGD(kNameFusionPass, "Start CheckNodeShapeSupported.");
  if (CheckNodeShapeSupported()) {
    OP_LOGI(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }

  /*
  * Create new Desc AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start CreateAttentionDesc.");
  FUSION_PASS_CHECK(CreateAttentionDesc() != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input desc for bsb failed."),
                    return FAILED);
  /*
  * Create new Node AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start Create new Node AttentionScore.");
  ge::NodePtr bsb_node = graph.AddNode(bsb_desc);
  FUSION_PASS_CHECK(bsb_node == nullptr,
                    OP_LOGE(kNameFusionPass, "Add AttentionScoreGrad to graph failed."),
                    return FAILED);
  /*
  * Add Edges for AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start AddEdgesForBsbNode.");
  FUSION_PASS_CHECK(AddEdgesForBsbNode(bsb_node) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add Edges For BsbNode failed."),
                    return FAILED);
  /*
  * Add Control Edges for AttentionScore.
  */
  OP_LOGD(kNameFusionPass, "Start AddControlEdgesForBsbNode.");
  FUSION_PASS_CHECK(AddControlEdgesForBsbNode(bsb_node) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add Edges For BsbNode failed."),
                    return FAILED);
  /*
  * Delete fused node in graph.
  */
  OP_LOGD(kNameFusionPass, "Start DeleteOldNode.");
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != DeleteOldNode(graph),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node failed"),
      return FAILED);

  return SUCCESS;
}

Status ZAttentionScoreGradFusionPass::AddOutputEdgeForNode(ge::NodePtr ori_node, ge::NodePtr new_node,
                                                           int unlinkIndex, int new_node_index) const {
  if (ori_node->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : ori_node->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(new_node_index), inAnchorPtr),
                        CUBE_CALL_ERR_REPORT(
                          kNameFusionPass,
                          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                          new_node->GetName().c_str(), 0, new_node->GetName().c_str(), 0),
                        return FAILED);
    }
  }
  return SUCCESS;
}


REGISTER_PASS("ZAttentionScoreGradFusionPass",
              SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              ZAttentionScoreGradFusionPass);
}  // namespace fe
