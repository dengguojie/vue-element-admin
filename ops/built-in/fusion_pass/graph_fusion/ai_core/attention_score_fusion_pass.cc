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
 * \file attention_score_fusion_pass.cc
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
#include "attention_score_fusion_pass.h"

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
static const char PATTERN_BATCHMATMUL[] = "batch_matmul";
static const char PATTERN_BATCHMATMUL2[] = "batch_matmul2";
static const char PATTERN_BATCHMATMUL3[] = "batch_matmul3";
static const char PATTERN_FUSEDMULADD[] = "fused_mul_add";
static const char PATTERN_SOFTMAXV2WITHDROPOUT[] = "softmax_v2_with_dropout";
static const char PATTERN_CONFUSIONTRANSPOSE[] = "confusion_transpose";
static const char PATTERN_SOFTMAXV2[] = "softmax_v2";

static const char BATCHMATMULV2[] = "BatchMatMulV2";
static const char FUSEDMULADD[] = "FusedMulAdd";
static const char SOFTMAXV2WITHDROPOUTDOMASKV3D[] = "SoftmaxV2WithDropOutDoMaskV3D";
static const char CONFUSIONTRANSPOSE[] = "ConfusionTransposeD";
static const char SOFTMAXV2[] = "SoftmaxV2";
static const int ALIGN_UNIT = 16;
static const int CONFUSION_DIM_ONE = 12288;
static const int CONFUSION_DIM_TWO = 768;
static const int NUM_TWO = 2;
static const int NUM_THREE = 3;
static const int NUM_FOUR = 4;
static const int NUM_FIVE = 5;
static const int NUM_SIX = 6;

static const char kNameFusionPass[] = "ZAttentionScoreFusionPass";
}  // namespace

vector<FusionPattern *> ZAttentionScoreFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern.");
  pattern->AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_FUSEDMULADD, {FUSEDMULADD})
    .AddOpDesc(PATTERN_SOFTMAXV2WITHDROPOUT, {SOFTMAXV2WITHDROPOUTDOMASKV3D})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_FUSEDMULADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXV2WITHDROPOUT, {PATTERN_FUSEDMULADD})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2WITHDROPOUT})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  FusionPattern *pattern1 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern1."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern1.");
  pattern1->AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_FUSEDMULADD, {FUSEDMULADD})
    .AddOpDesc(PATTERN_SOFTMAXV2, {SOFTMAXV2})
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {CONFUSIONTRANSPOSE})
    .SetInputs(PATTERN_FUSEDMULADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXV2, {PATTERN_FUSEDMULADD})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2})
    .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_BATCHMATMUL2})
    .SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern1);
  OP_LOGD(kNameFusionPass, "End to define pattern1.");

  return patterns;
}

Status ZAttentionScoreFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGI(kNameFusionPass, "Start ZAttentionScoreFusionPass.");

  batch_matmul_node1 = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(batch_matmul_node1 == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get batch_matmul_node1 not success."),
                    return PARAM_INVALID);

  ge::NodePtr output_batch_matmul_node1 = GetNodeFromMapping(PATTERN_BATCHMATMUL2, mapping);
  FUSION_PASS_CHECK(output_batch_matmul_node1 == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get batch_matmul_node2 not success."),
                    return PARAM_INVALID);

  ge::NodePtr fused_mul_add_node = GetNodeFromMapping(PATTERN_FUSEDMULADD, mapping);
  FUSION_PASS_CHECK(fused_mul_add_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get fused_mul_add_node not success."),
                    return PARAM_INVALID);

  ge::NodePtr confusion_transpose_node = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE, mapping);
  FUSION_PASS_CHECK(confusion_transpose_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get confusion_transpose_node not success."),
                    return PARAM_INVALID);

  vector<int64_t> confusion_transpose_out_dims = confusion_transpose_node->GetOpDesc()->
                                                       MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(confusion_transpose_out_dims.empty(),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get confusion_transpose_out_dims not success."),
                    return PARAM_INVALID);
  batch_matmul_node2 = output_batch_matmul_node1;
  std::string batch_matmul2_name = batch_matmul_node2->GetName().c_str();
  std::string tar_name = "gradient_tape";
  size_t found = batch_matmul2_name.find(tar_name);
  if (found != std::string::npos) {
    OP_LOGI(kNameFusionPass, "Graph not change.");
    return NOT_CHANGED;
  }

  vector<int64_t> batch_matmul_node2_dims = batch_matmul_node2->GetOpDesc()->MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> batch_matmul_node1_input2_dims = batch_matmul_node1->GetOpDesc()->
                                                      MutableInputDesc(1)->GetShape().GetDims();
  vector<int64_t> batch_matmul_node1_dims = batch_matmul_node1->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims();
  if (batch_matmul_node1_dims != batch_matmul_node1_input2_dims || batch_matmul_node1_dims != batch_matmul_node2_dims) {
    return NOT_CHANGED;
  }
  OP_LOGD(kNameFusionPass, "Check BatchMatMul node input dims.");
  bool traning = true;
  softmax_node = GetNodeFromMapping(PATTERN_SOFTMAXV2WITHDROPOUT, mapping);
  if (softmax_node == nullptr) {
    softmax_node = GetNodeFromMapping(PATTERN_SOFTMAXV2, mapping);
    FUSION_PASS_CHECK(softmax_node == nullptr,
                      CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get softmax_node not success."),
                      return PARAM_INVALID);
    traning = false;
  } else {
    OP_LOGI(kNameFusionPass, "Not Support traing, graph not change.");
    return NOT_CHANGED;
  }
  if (traning) {
    for (auto netout_in_data_anchor : softmax_node->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      ge::NodePtr temp_batch_node = netout_in_data_anchor->GetOwnerNode();
      if (temp_batch_node->GetName().c_str() != batch_matmul_node2->GetName().c_str()) {
        batch_matmul_node3 = temp_batch_node;
        break;
      }
    }
  }
  OP_LOGD(kNameFusionPass, "Check the training net.");
  vector<int64_t> softmax_shape_dims = softmax_node->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims();
  FUSION_PASS_CHECK(softmax_shape_dims.empty(),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get softmax_dims not success."),
                    return PARAM_INVALID);
  auto dim_size = softmax_shape_dims.size();
  if (dim_size != NUM_FOUR && dim_size != NUM_SIX) {
    OP_LOGI(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }
  int64_t batch_dim0 = softmax_shape_dims[0];
  int64_t batch_dim1 = softmax_shape_dims[1];
  int64_t batch_dim2 = softmax_shape_dims[NUM_TWO];
  int64_t batch_dim3 = softmax_shape_dims[NUM_THREE];
  bool shape_not_matched = false;
  if (batch_dim1 != ALIGN_UNIT || batch_dim2 != batch_dim3) {
    shape_not_matched = true;
  }
  auto softmax_format = softmax_node->GetOpDesc()->GetInputDesc(0).GetFormat();
  if (dim_size == NUM_FOUR) {
    if (batch_dim0 * batch_dim2 != CONFUSION_DIM_ONE) {
      shape_not_matched = true;
    }
  } else if (dim_size == NUM_SIX && softmax_format == ge::FORMAT_FRACTAL_NZ) {
    if (batch_dim0 * batch_dim3 != CONFUSION_DIM_TWO) {
      shape_not_matched = true;
    }
  } else {
    shape_not_matched = true;
  }

  if (shape_not_matched) {
    OP_LOGI(kNameFusionPass, "Shape not match, graph not change.");
    return NOT_CHANGED;
  }
  std::shared_ptr<ge::OpDesc> bsb_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    bsb_desc = std::make_shared<ge::OpDesc>(softmax_node->GetName() + "/AttentionScore",
                                            "AttentionScore"),
    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("query",
                                           *(batch_matmul_node1->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input0 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("key",
                                           *(batch_matmul_node1->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input1 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("value",
                                           *(batch_matmul_node2->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input2 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("padding_mask",
                                           *(fused_mul_add_node->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input3 of AttentionScore failed."),
                    return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("scale",
                                           *(fused_mul_add_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input4 of AttentionScore failed."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(bsb_desc->AddInputDesc("drop_mask",
                                             *(softmax_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
  } else {
    FUSION_PASS_CHECK(bsb_desc->AddInputDesc("drop_mask",
                                             *(fused_mul_add_node->GetOpDesc()->MutableInputDesc(NUM_TWO))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input5 of AttentionScore failed."),
                      return FAILED);
  }

  FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("attention_score",
                                             *(confusion_transpose_node->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of AttentionScore failed."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("softmax_output",
                                             *(softmax_node->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add output1 of AttentionScore failed."),
                      return FAILED);
  }
  bool first_transpose_a = false;
  bool first_transpose_b = false;
  bool second_transpose_a = false;
  bool second_transpose_b = false;
  float keep_prob = 0;
  vector<int64_t> axes = {};
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batch_matmul_node1->GetOpDesc(), "adj_x1", first_transpose_a),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batch_matmul_node1->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batch_matmul_node1->GetOpDesc(), "adj_x2", first_transpose_b),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x2 of %s.", batch_matmul_node1->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batch_matmul_node2->GetOpDesc(), "adj_x1", second_transpose_a),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x1 of %s.", batch_matmul_node2->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batch_matmul_node2->GetOpDesc(), "adj_x2", second_transpose_b),
    OP_LOGE(kNameFusionPass, "Failed to get adj_x2 of %s.", batch_matmul_node2->GetName().c_str()),
    return FAILED);
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetListInt(softmax_node->GetOpDesc(), "axes", axes),
    OP_LOGE(kNameFusionPass, "Failed to get axes of %s.", softmax_node->GetName().c_str()),
    return FAILED);
  if (traning) {
      FUSION_PASS_CHECK(
        !ge::AttrUtils::GetFloat(softmax_node->GetOpDesc(), "keep_prob", keep_prob),
        OP_LOGE(kNameFusionPass, "Failed to get keep_prod of %s.", softmax_node->GetName().c_str()),
        return FAILED);

      ge::AttrUtils::SetBool(bsb_desc, "keep_prob", keep_prob);
  }
  ge::AttrUtils::SetBool(bsb_desc, "query_transpose", first_transpose_a);
  ge::AttrUtils::SetBool(bsb_desc, "key_transpose", first_transpose_b);
  ge::AttrUtils::SetBool(bsb_desc, "bmm_score_transpose_a", second_transpose_a);
  ge::AttrUtils::SetBool(bsb_desc, "bmm_score_transpose_b", second_transpose_b);
  ge::AttrUtils::SetListInt(bsb_desc, "softmax_axes", axes);

  ge::NodePtr bsb_node = graph.AddNode(bsb_desc);
  FUSION_PASS_CHECK(bsb_node == nullptr,
                    OP_LOGE(kNameFusionPass, "Add AttentionScore to graph failed."),
                    return FAILED);

  // add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node1->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node1->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batch_matmul_node2->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_TWO)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_mul_add_node->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_THREE)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_mul_add_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(NUM_FOUR)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                    return FAILED);
  if (traning) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmax_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_mul_add_node->GetInDataAnchor(NUM_TWO)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(NUM_FIVE)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."),
                      return FAILED);
  }
  if (traning) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmax_node->GetOutDataAnchor(1),
                                                 batch_matmul_node3->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to remove edge softmax to bmm3."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bsb_node->GetOutDataAnchor(1),
                                              batch_matmul_node3->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to add edge bsbNode to bmm3."),
                      return FAILED);
  }
  AddOutputEdgeForNode(confusion_transpose_node, bsb_node, 0, 0);
  if (traning) {
    AddOutputEdgeForNode(softmax_node, bsb_node, 0, 1);
  }

  // add control edge
  if (softmax_node->GetInControlAnchor()) {
    for (auto out_control_anchor : softmax_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, bsb_node->GetInControlAnchor()) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node input control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_control_anchor, softmax_node->GetInControlAnchor()) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass,
                                                       "Remove softmax_node node input control edge failed."),
                        return FAILED);
    }
  }
  if (softmax_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : softmax_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmax_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
                        CUBE_CALL_ERR_REPORT(kNameFusionPass, "Remove softmax node out control edge failed."),
                        return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(bsb_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          CUBE_CALL_ERR_REPORT(kNameFusionPass, "Add new node out control edge failed."), return FAILED);
    }
  }
  // delete edge
  // delete node
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node1),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node1->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fused_mul_add_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     fused_mul_add_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(batch_matmul_node2),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     batch_matmul_node2->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(softmax_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     softmax_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(confusion_transpose_node),
      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                     confusion_transpose_node->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

Status ZAttentionScoreFusionPass::AddOutputEdgeForNode(ge::NodePtr ori_node, ge::NodePtr new_node,
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


REGISTER_PASS("ZAttentionScoreFusionPass",
              BUILT_IN_GRAPH_PASS,
              ZAttentionScoreFusionPass);
}  // namespace fe
