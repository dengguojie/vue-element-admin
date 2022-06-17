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
 * \file layernorm_training_fusion_pass.cc
 * \brief fuse matched scope to target LayerNorm and LayerNormGrad.
 */
#include "layernorm_training_fusion_pass.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "op_const.h"
#include "op_attr.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const std::string ADD = "Add";
static const std::string ADD_N = "AddN";
static const std::string MUL = "Mul";
static const std::string SUB = "Sub";
static const std::string RSQRT = "Rsqrt";
static const std::string NEG = "Neg";
static const std::string RESHAPE = "Reshape";
static const std::string REDUCE_MEAN = "ReduceMean";
static const std::string REDUCE_SUM = "ReduceSum";
static const std::string REDUCE_SUM_D = "ReduceSumD";
static const std::string REAL_DIV = "RealDiv";
static const std::string SQUARED_DIFFERENCE = "SquaredDifference";
static const std::string RSQRT_GRAD = "RsqrtGrad";
static const char* PATTERN_ADD_1 = "add_1";
static const char* PATTERN_ADD_2 = "add_2";
static const char* PATTERN_MUL_1 = "mul_1";
static const char* PATTERN_MUL_2 = "mul_2";
static const char* PATTERN_MUL_3 = "mul_3";
static const char* PATTERN_SUB = "sub";
static const char* PATTERN_RSQRT = "rsqrt";
static const char* PATTERN_REDUCE_MEAN_1 = "reduce_mean_1";
static const char* PATTERN_REDUCE_MEAN_2 = "reduce_mean_2";
static const char* PATTERN_SQUARED_DIFFERENCE = "squared_difference";
static const char* PATTERN_GRAD_MUL_7 = "grad_mul_7";
static const char* PATTERN_RSQRT_GRAD = "rsqrt_grad";
static const char* PATTERN_GRAD_SUB = "grad_sub";
static const char* PATTERN_ADD_N_1 = "add_n_1";  // concat to output0
static const uint32_t ADD_N_1_INPUT_SIZE = 3;
static const size_t ADD_N_0_IN_OUT_SIZE = 2;
static const std::string FUSED_OP_TYPE_GRAD = "LayerNormGrad";

vector<FusionPattern*> LayerNormTrainingFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LayerNormTrainingFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new an object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_REDUCE_MEAN_1, {REDUCE_MEAN})
          .AddOpDesc(PATTERN_SQUARED_DIFFERENCE, {SQUARED_DIFFERENCE})
          .AddOpDesc(PATTERN_REDUCE_MEAN_2, {REDUCE_MEAN})
          .AddOpDesc(PATTERN_ADD_1, {ADD})
          .AddOpDesc(PATTERN_RSQRT, {RSQRT})
          .AddOpDesc(PATTERN_MUL_1, {MUL})
          .AddOpDesc(PATTERN_MUL_2, {MUL})
          .AddOpDesc(PATTERN_MUL_3, {MUL})
          .AddOpDesc(PATTERN_SUB, {SUB})
          .AddOpDesc(PATTERN_ADD_2, {ADD})
          .AddOpDesc(PATTERN_GRAD_MUL_7, {MUL})
          .AddOpDesc(PATTERN_RSQRT_GRAD, {RSQRT_GRAD})
          .AddOpDesc(PATTERN_GRAD_SUB, {SUB})
          .AddOpDesc(PATTERN_ADD_N_1, {ADD_N})
          .SetInputs(PATTERN_SQUARED_DIFFERENCE, {PATTERN_REDUCE_MEAN_1})
          .SetInputs(PATTERN_REDUCE_MEAN_2, {PATTERN_SQUARED_DIFFERENCE})
          .SetInputs(PATTERN_ADD_1, {PATTERN_REDUCE_MEAN_2})
          .SetInputs(PATTERN_RSQRT, {PATTERN_ADD_1})
          .SetInputs(PATTERN_MUL_1, {PATTERN_RSQRT})
          .SetInputs(PATTERN_MUL_2, {PATTERN_REDUCE_MEAN_1, PATTERN_MUL_1})
          .SetInputs(PATTERN_MUL_3, {PATTERN_MUL_1})
          .SetInputs(PATTERN_SUB, {PATTERN_MUL_2})
          .SetInputs(PATTERN_ADD_2, {PATTERN_MUL_3, PATTERN_SUB})
          .SetOutputs(PATTERN_RSQRT, {{0, PATTERN_RSQRT_GRAD}}, false)
          .SetOutputs(PATTERN_REDUCE_MEAN_1, {{0, PATTERN_GRAD_SUB}}, false)
          .SetOutputs(PATTERN_GRAD_SUB, {{0, PATTERN_GRAD_MUL_7}})
          .SetOutputs(PATTERN_GRAD_MUL_7, {{0, PATTERN_ADD_N_1}})
          .SetOutput(PATTERN_ADD_2);
  patterns.emplace_back(pattern);

  return patterns;
}

Status LayerNormTrainingFusionPass::CheckNullPtr(const std::vector<ge::NodePtr>& pattern_nodes) {
  for (const auto& node_ptr : pattern_nodes) {
    if (node_ptr == nullptr || node_ptr->GetOpDesc() == nullptr) {
      return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::CheckLNGNodes(const ge::NodePtr& add_n_node_1, const ge::NodePtr& mul_node_7,
                                                  const ge::NodePtr& mul_node_1, const uint32_t addn_input_size) {
  bool addn_is_ok = true;
  std::vector<std::string> true_div_type = {MUL, REAL_DIV};

  if (addn_input_size < ADD_N_1_INPUT_SIZE) {
    addn_is_ok = false;
  } else {
    uint32_t mul_div_size = 0;
    for (uint32_t idx = 0; idx < addn_input_size; idx++) {
      auto tmp_input_node = FusionTurbo::GetPeerOutNode(add_n_node_1, idx);
      FUSION_PASS_CHECK(tmp_input_node == nullptr,
                        OP_LOGD(FUSED_OP_TYPE, "add_n1 input node is null."), return NOT_CHANGED);
      auto tmp_input_op = tmp_input_node->GetType();
      if (std::find(true_div_type.begin(), true_div_type.end(), tmp_input_op) != true_div_type.end()) {
        mul_div_size += 1;
      }
    }

    if (mul_div_size < ADD_N_1_INPUT_SIZE) {
      addn_is_ok = false;
    }
  }
  FUSION_PASS_CHECK(!addn_is_ok,
                    OP_LOGD(FUSED_OP_TYPE, "add_n1's input count is mismatched."), return NOT_CHANGED);

  uint32_t r2nd_idx = 2;
  FUSION_PASS_CHECK(FusionTurbo::GetPeerOutNode(add_n_node_1, addn_input_size - r2nd_idx) != mul_node_7,
                    OP_LOGD(FUSED_OP_TYPE, "add_n1's last but one input is mismatched."), return NOT_CHANGED);

  uint32_t r3rd_idx = 3;
  FUSION_PASS_CHECK(FusionTurbo::GetPeerOutNode(add_n_node_1, addn_input_size - r3rd_idx) != mul_node_1,
                    OP_LOGD(FUSED_OP_TYPE, "add_n1's last but two input is mismatched."), return NOT_CHANGED);

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::GetReduceOpAttr(std::vector<int64_t>& axes,
                                                    bool& keep_dims, const ge::NodePtr& node) {
  size_t input_idx_1 = 1;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  if (!ops::GetConstIntData(op, input_idx_1, axes)) {
    return NOT_CHANGED;
  }

  const bool default_keep_dims = false;
  const std::pair<int64_t, std::string> axis_attr_info{0, "keep_dims"};
  if (!ops::GetAttrValue(op, axis_attr_info, keep_dims, default_keep_dims)) {
    return NOT_CHANGED;
  }

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::CheckReduceOpAttr(const std::vector<int64_t>& axes_1,
                                                      const bool keep_dims_1,
                                                      const ge::NodePtr& mean_node_1,
                                                      const ge::NodePtr& mean_node_2) {
  std::vector<int64_t> axes_2 = {-1};
  bool keep_dims_2 = false;
  FUSION_PASS_CHECK(GetReduceOpAttr(axes_2, keep_dims_2, mean_node_2) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get attributes of mean2."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!(axes_1.size() == 1 && axes_1 == axes_2),
                    OP_LOGD(FUSED_OP_TYPE, "the axes of mean nodes are not same."), return NOT_CHANGED);

  std::vector<bool> keep_dims = {keep_dims_1, keep_dims_2};
  FUSION_PASS_CHECK(std::find(keep_dims.begin(), keep_dims.end(), false) != keep_dims.end(),
                    OP_LOGD(FUSED_OP_TYPE, "the keep_dims of mean is false."), return NOT_CHANGED);

  auto input_desc = mean_node_1->GetOpDesc()->MutableInputDesc(0);
  FUSION_PASS_CHECK(input_desc == nullptr, OP_LOGD(FUSED_OP_TYPE, "mean1's input is nullptr."), return NOT_CHANGED);
  size_t dims_size = input_desc->MutableShape().GetDims().size();
  FUSION_PASS_CHECK(dims_size < 1,
                    OP_LOGD(FUSED_OP_TYPE, "input shape should be greater than 0."), return NOT_CHANGED);
  FUSION_PASS_CHECK(axes_1[0] != -1 && axes_1[0] != static_cast<int64_t>(dims_size - 1),
                    OP_LOGD(FUSED_OP_TYPE, "the axes is not the last axis."), return NOT_CHANGED);

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::GetInput24PeerInNodes(const ge::NodePtr& grad_rsqrt_node,
                                                          const ge::NodePtr& rsqrt_node,
                                                          ge::NodePtr& grad_mul_node_5,
                                                          ge::NodePtr& add_n_node_0,
                                                          ge::NodePtr& grad_mul_node_8) {
  auto reshape_node_1 = FusionTurbo::GetPeerOutNode(grad_rsqrt_node, 1);
  FUSION_PASS_CHECK(reshape_node_1 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "reshape node is null."), return NOT_CHANGED);
  if (reshape_node_1 == rsqrt_node) {
    reshape_node_1 = FusionTurbo::GetPeerOutNode(grad_rsqrt_node, 0);
    FUSION_PASS_CHECK(reshape_node_1 == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "reshape node is null."), return NOT_CHANGED);
  }
  auto reduce_sum_node_1 = FusionTurbo::GetPeerOutNode(reshape_node_1, 0);
  FUSION_PASS_CHECK(reduce_sum_node_1 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "reduce_sum node is null."), return NOT_CHANGED);
  grad_mul_node_5 = FusionTurbo::GetPeerOutNode(reduce_sum_node_1, 0);
  FUSION_PASS_CHECK(grad_mul_node_5 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "mul5 node is null."), return NOT_CHANGED);

  add_n_node_0 = FusionTurbo::GetPeerOutNode(grad_mul_node_5, 0);
  FUSION_PASS_CHECK(add_n_node_0 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "add_n0 node is null."), return NOT_CHANGED);
  if (add_n_node_0->GetType() != ADD_N) {
    add_n_node_0 = FusionTurbo::GetPeerOutNode(grad_mul_node_5, 1);
    FUSION_PASS_CHECK(add_n_node_0 == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "add_n0 node is null."), return NOT_CHANGED);
  }

  std::vector<ge::NodePtr> peer_in_nodes = FusionTurbo::GetPeerInNodes(add_n_node_0, 0);
  FUSION_PASS_CHECK(peer_in_nodes.size() != ADD_N_0_IN_OUT_SIZE,
                    OP_LOGD(FUSED_OP_TYPE, "add_n0's output count is not correct."), return NOT_CHANGED);
  for (const auto& node : peer_in_nodes) {
    FUSION_PASS_CHECK(node == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "node is null."), return NOT_CHANGED);
    if (node != grad_mul_node_5) {
      grad_mul_node_8 = node;
      break;
    }
  }

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::GetInput13PeerInNodes(const ge::NodePtr& add_n_node_0,
                                                          const ge::NodePtr& grad_sub_node,
                                                          ge::NodePtr& grad_mul_node_4,
                                                          ge::NodePtr& grad_mul_4_up_node,
                                                          ge::NodePtr& grad_mul_node_2){
  FUSION_PASS_CHECK(add_n_node_0->GetOpDesc()->GetInputsSize() != ADD_N_0_IN_OUT_SIZE,
                    OP_LOGD(FUSED_OP_TYPE, "add_n0's input count is not correct."), return NOT_CHANGED);

  auto input_node_1 = FusionTurbo::GetPeerOutNode(grad_sub_node, 0);
  FUSION_PASS_CHECK(input_node_1 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "input node1 is null."), return NOT_CHANGED);
  auto input_node_3 = FusionTurbo::GetPeerOutNode(grad_sub_node, 1);
  FUSION_PASS_CHECK(input_node_3 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "input node3 is null."), return NOT_CHANGED);

  grad_mul_node_2 = FusionTurbo::GetPeerOutNode(add_n_node_0, 0);
  FUSION_PASS_CHECK(grad_mul_node_2 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "mul2 is null."), return NOT_CHANGED);
  bool mul_2_is_ok = false;
  for (const auto& node : FusionTurbo::GetPeerInNodes(input_node_1, 0)) {
    FUSION_PASS_CHECK(node == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "node is null."), return NOT_CHANGED);
    if (node == grad_mul_node_2) {
      mul_2_is_ok = true;
      break;
    }
  }
  if (!mul_2_is_ok){
    grad_mul_node_2 = FusionTurbo::GetPeerOutNode(add_n_node_0, 1);
    FUSION_PASS_CHECK(grad_mul_node_2 == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "mul2 is null."), return NOT_CHANGED);
  }

  grad_mul_node_4 = FusionTurbo::GetPeerOutNode(add_n_node_0, 0);
  if (grad_mul_node_4 == grad_mul_node_2) {
    grad_mul_node_4 = FusionTurbo::GetPeerOutNode(add_n_node_0, 1);
    FUSION_PASS_CHECK(grad_mul_node_4 == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "mul4 is null."), return NOT_CHANGED);
  }

  grad_mul_4_up_node = FusionTurbo::GetPeerOutNode(grad_mul_node_4, 0);
  FUSION_PASS_CHECK(grad_mul_4_up_node == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "mul4 input node is null."), return NOT_CHANGED);
  if (grad_mul_4_up_node == input_node_3) {
    grad_mul_4_up_node = FusionTurbo::GetPeerOutNode(grad_mul_node_4, 1);
    FUSION_PASS_CHECK(grad_mul_4_up_node == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "mul4 input node is null."), return NOT_CHANGED);
  }

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::GetInput0PeerInNodes(const ge::NodePtr& grad_mul_node_2,
                                                         const ge::NodePtr& grad_sub_node,
                                                         ge::NodePtr& sum_node_2,
                                                         ge::NodePtr& grad_mul_node_1) {
  auto input_node_1 = FusionTurbo::GetPeerOutNode(grad_sub_node, 0);
  FUSION_PASS_CHECK(input_node_1 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "input node1 is null."), return NOT_CHANGED);
  auto input_node_0 = FusionTurbo::GetPeerOutNode(grad_mul_node_2, 0);
  FUSION_PASS_CHECK(input_node_0 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "input node1 is null."), return NOT_CHANGED);
  auto out_idx = grad_mul_node_2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx();
  if (input_node_0 == input_node_1) {
    input_node_0 = FusionTurbo::GetPeerOutNode(grad_mul_node_2, 1);
    FUSION_PASS_CHECK(input_node_0 == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "input node1 is null."), return NOT_CHANGED);
    out_idx = grad_mul_node_2->GetInDataAnchor(1)->GetPeerOutAnchor()->GetIdx();
  }

  std::vector<ge::NodePtr> peer_in_nodes = FusionTurbo::GetPeerInNodes(input_node_0, out_idx);
  size_t peer_in_size = 4;
  FUSION_PASS_CHECK(peer_in_nodes.size() != peer_in_size,
                    OP_LOGD(FUSED_OP_TYPE, "input0's peer in nodes count is mismatched."), return NOT_CHANGED);

  ge::NodePtr grad_neg_node = nullptr;
  for (const auto& tmp_node : peer_in_nodes) {
    FUSION_PASS_CHECK(tmp_node == nullptr,
                      OP_LOGD(FUSED_OP_TYPE, "input0's peer in node is nullptr."), return NOT_CHANGED);
    auto tmp_node_type = tmp_node->GetType();
    if (tmp_node == grad_mul_node_2) {
      continue;
    }

    if (tmp_node_type == REDUCE_SUM || tmp_node_type == REDUCE_SUM_D) {
      sum_node_2 = tmp_node;
    } else if (tmp_node_type == NEG) {
      grad_neg_node = tmp_node;
    } else if (tmp_node_type == MUL) {
      grad_mul_node_1 = tmp_node;
    } else {
      return NOT_CHANGED;
    }
  }

  std::vector<ge::NodePtr> tmp_vec = {sum_node_2, grad_mul_node_1, grad_neg_node};
  FUSION_PASS_CHECK(std::find(tmp_vec.begin(), tmp_vec.end(), nullptr) != tmp_vec.end(),
                    OP_LOGD(FUSED_OP_TYPE, "input0's peer in node is nullptr."), return NOT_CHANGED);

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::AddTargetNode(FusionTurbo& turbo_instance,
                                                  const ge::NodePtr& mean_node_1,
                                                  const ge::NodePtr& rsqrt_grad_node,
                                                  ge::NodePtr& layer_norm_node,
                                                  ge::NodePtr& layer_norm_grad_node) {
  std::string layer_norm_name = mean_node_1->GetOpDesc()->GetName() + "/" + FUSED_OP_TYPE;
  layer_norm_node = turbo_instance.AddNodeOnly(layer_norm_name, FUSED_OP_TYPE);
  FUSION_PASS_CHECK(layer_norm_node == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "LayerNorm node is null, do nothing."), return NOT_CHANGED);

  std::string layer_norm_grad_name = rsqrt_grad_node->GetOpDesc()->GetName() + "/" + FUSED_OP_TYPE_GRAD;
  layer_norm_grad_node = turbo_instance.AddNodeOnly(layer_norm_grad_name, FUSED_OP_TYPE_GRAD);
  FUSION_PASS_CHECK(layer_norm_grad_node == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "LayerNormGrad node is null, do nothing."), return NOT_CHANGED);

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::SetLayerNormAttr(const ge::NodePtr& node,
                                                     const std::vector<int64_t>& axes_vec, const ge::NodePtr& add_1) {
  auto add_op = ge::OpDescUtils::CreateOperatorFromNode(add_1);
  std::vector<float> epsilon_vec = {1e-7};
  if (!ops::GetConstIntData(add_op, 0, epsilon_vec)) {
    if (!ops::GetConstIntData(add_op, 1, epsilon_vec)) {
      OP_LOGD(FUSED_OP_TYPE, "set epsilon to default value.");
    }
  }

  auto node_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(node_desc == nullptr, OP_LOGD(FUSED_OP_TYPE, "OpDesc is null, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(node_desc, "begin_norm_axis", axes_vec[0]),
                    OP_LOGD(FUSED_OP_TYPE, "failed to set begin_norm_axis, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(node_desc, "begin_params_axis", -1),
                    OP_LOGD(FUSED_OP_TYPE, "failed to set begin_params_axis, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(node_desc, "epsilon", epsilon_vec[0]),
                    OP_LOGD(FUSED_OP_TYPE, "failed to set epsilon, do nothing."), return NOT_CHANGED);

  return SUCCESS;
}

void LayerNormTrainingFusionPass::GetInputRelations(Relations& input_relations,
                                                    const std::vector<std::pair<ge::NodePtr, ge::NodePtr>>& pairs) {
  for (size_t idx = 0; idx < pairs.size(); idx++) {
    auto nodes = pairs[idx];
    int32_t input_idx = 1;
    if (FusionTurbo::GetPeerOutNode(nodes.first, 0) != nodes.second) {
      input_idx = 0;
    }
    NodeIndex node_index(nodes.first, input_idx, PEER);
    std::vector<NodeIndex> node_index_vec = {node_index};
    input_relations.Add(static_cast<int32_t>(idx), node_index_vec);
  }
}

void LayerNormTrainingFusionPass::GetOutputRelations(Relations& output_relations,
                                                     const std::vector<ge::NodePtr>& nodes) {
  for (size_t idx = 0; idx < nodes.size(); idx++) {
    NodeIndex node_index(nodes[idx], 0, PEER);
    std::vector<NodeIndex> node_index_vec = {node_index};
    output_relations.Add(static_cast<int32_t>(idx), node_index_vec);
  }
}

Status LayerNormTrainingFusionPass::UpdateAddN(FusionTurbo& turbo_instance, const ge::NodePtr& layer_norm_grad_node,
                                               const ge::NodePtr& add_n_node_1, const uint32_t input_size) {
  auto add_n_desc = add_n_node_1->GetOpDesc();
  uint32_t deleted_size = 2;
  // delete last two inputs
  std::vector<int32_t> input_idx = {static_cast<int32_t>(input_size - deleted_size),
                                    static_cast<int32_t>(input_size - 1)};
  FUSION_PASS_CHECK(turbo_instance.BreakInput(add_n_node_1, input_idx) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to break input edges, do nothing."),
                    return FAILED);
  for (uint32_t idx = input_size - 1; idx >= input_size - deleted_size; idx--) {
    OpDescUtils::ClearInputDesc(add_n_desc, idx);
  }
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(add_n_desc, "N", input_size - deleted_size),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set attribute N failed, do nothing."),
                    return FAILED);

  // only for second input is LayerNormGrad
  auto grad_op = ge::OpDescUtils::CreateOperatorFromNode(layer_norm_grad_node);
  FUSION_PASS_CHECK(grad_op.InferShapeAndType() != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to run LayerNormGrad's InferShape."),
                    return FAILED);
  FUSION_PASS_CHECK(turbo_instance.UpdateInputByPeer(add_n_node_1, 1, layer_norm_grad_node, 0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to update AddN's input."), return FAILED);

  return SUCCESS;
}

Status LayerNormTrainingFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  ge::NodePtr mean_node_1 = GetNodeFromMapping(PATTERN_REDUCE_MEAN_1, mapping);
  ge::NodePtr squared_difference_node = GetNodeFromMapping(PATTERN_SQUARED_DIFFERENCE, mapping);
  ge::NodePtr mean_node_2 = GetNodeFromMapping(PATTERN_REDUCE_MEAN_2, mapping);
  ge::NodePtr add_node_1 = GetNodeFromMapping(PATTERN_ADD_1, mapping);
  ge::NodePtr rsqrt_node = GetNodeFromMapping(PATTERN_RSQRT, mapping);
  ge::NodePtr mul_node_1 = GetNodeFromMapping(PATTERN_MUL_1, mapping);
  ge::NodePtr mul_node_2 = GetNodeFromMapping(PATTERN_MUL_2, mapping);
  ge::NodePtr mul_node_3 = GetNodeFromMapping(PATTERN_MUL_3, mapping);
  ge::NodePtr sub_node = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr add_node_2 = GetNodeFromMapping(PATTERN_ADD_2, mapping);

  ge::NodePtr grad_mul_node_7 = GetNodeFromMapping(PATTERN_GRAD_MUL_7, mapping);
  ge::NodePtr grad_rsqrt_node = GetNodeFromMapping(PATTERN_RSQRT_GRAD, mapping);
  ge::NodePtr grad_sub_node = GetNodeFromMapping(PATTERN_GRAD_SUB, mapping);
  ge::NodePtr add_n_node_1 = GetNodeFromMapping(PATTERN_ADD_N_1, mapping);

  std::vector<ge::NodePtr> pattern_nodes = {mean_node_1, squared_difference_node, mean_node_2, add_node_1,
                                            rsqrt_node, mul_node_1, mul_node_2, mul_node_3, sub_node, add_node_2,
                                            grad_mul_node_7, grad_rsqrt_node, grad_sub_node, add_n_node_1};
  FUSION_PASS_CHECK(CheckNullPtr(pattern_nodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "node has nullptr, please check."), return NOT_CHANGED);

  std::vector<int64_t> axes_1 = {-1};
  bool keep_dims_1 = false;
  FUSION_PASS_CHECK(GetReduceOpAttr(axes_1, keep_dims_1, mean_node_1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get attribute of mean1."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReduceOpAttr(axes_1, keep_dims_1, mean_node_1, mean_node_2) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to match attribute of mean1."), return NOT_CHANGED);

  // get input2, input4 peer out nodes for input relations
  ge::NodePtr grad_mul_node_5 = nullptr;
  ge::NodePtr add_n_node_0 = nullptr;
  ge::NodePtr grad_mul_node_8 = nullptr;
  Status ret_24 = GetInput24PeerInNodes(grad_rsqrt_node, rsqrt_node, grad_mul_node_5, add_n_node_0, grad_mul_node_8);
  FUSION_PASS_CHECK(ret_24 != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get input2 and input4 peer in nodes."), return NOT_CHANGED);
  // get input1, input3 peer out nodes for input relations
  ge::NodePtr grad_mul_node_4 = nullptr;
  ge::NodePtr grad_mul_4_up_node = nullptr;
  ge::NodePtr grad_mul_node_2 = nullptr;
  Status ret_13 = GetInput13PeerInNodes(add_n_node_0, grad_sub_node,
                                        grad_mul_node_4, grad_mul_4_up_node, grad_mul_node_2);
  FUSION_PASS_CHECK(ret_13 != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get input1 and input3 peer in nodes."), return NOT_CHANGED);
  // get input0 peer out nodes for input relations
  ge::NodePtr sum_node_2 = nullptr;
  ge::NodePtr grad_mul_node_1 = nullptr;
  Status ret_0 = GetInput0PeerInNodes(grad_mul_node_2, grad_sub_node, sum_node_2, grad_mul_node_1);
  FUSION_PASS_CHECK(ret_0 != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get input0 peer in nodes."), return NOT_CHANGED);
  // get output1 node
  auto sum_node_3 = FusionTurbo::GetPeerInNodes(grad_mul_node_8, 0).at(0);
  FUSION_PASS_CHECK(sum_node_3 == nullptr, OP_LOGD(FUSED_OP_TYPE, "failed to get output1 node."), return NOT_CHANGED);

  uint32_t input_size = add_n_node_1->GetOpDesc()->GetInputsSize();
  FUSION_PASS_CHECK(CheckLNGNodes(add_n_node_1, grad_mul_node_7, grad_mul_node_1, input_size) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "add_n1 is mismatched, do nothing."), return NOT_CHANGED);

  // add layernorm, layernormgrad into graph
  FusionTurbo turbo_instance(graph);
  ge::NodePtr layer_norm_node = nullptr;
  ge::NodePtr layer_norm_grad_node = nullptr;
  Status turbo_ret = AddTargetNode(turbo_instance, mean_node_1, grad_rsqrt_node,
                                   layer_norm_node, layer_norm_grad_node);
  FUSION_PASS_CHECK(turbo_ret != SUCCESS, OP_LOGD(FUSED_OP_TYPE, "failed to add target node."), return NOT_CHANGED);
  FUSION_PASS_CHECK(SetLayerNormAttr(layer_norm_node, axes_1, add_node_1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to set LayerNorm's attributes."), return NOT_CHANGED);

  // to be LayerNormGrad
  std::pair<ge::NodePtr, ge::NodePtr> grad_input_pair_0(sum_node_2, sum_node_2);
  std::pair<ge::NodePtr, ge::NodePtr> grad_input_pair_1(grad_sub_node, mean_node_1);
  std::pair<ge::NodePtr, ge::NodePtr> grad_input_pair_2(grad_mul_node_8, add_n_node_0);
  std::pair<ge::NodePtr, ge::NodePtr> grad_input_pair_3(grad_mul_node_4, grad_mul_4_up_node);
  std::pair<ge::NodePtr, ge::NodePtr> grad_input_pair_4(grad_mul_node_5, add_n_node_0);
  std::vector<std::pair<ge::NodePtr, ge::NodePtr>> grad_input_node_pairs = {grad_input_pair_0, grad_input_pair_1,
                                                                            grad_input_pair_2, grad_input_pair_3,
                                                                            grad_input_pair_4};
  std::vector<ge::NodePtr> grad_output_nodes = {add_n_node_1, sum_node_3, sum_node_2};
  Relations grad_input_relations;
  Relations grad_output_relations;
  GetInputRelations(grad_input_relations, grad_input_node_pairs);
  if (input_size > ADD_N_1_INPUT_SIZE) {  // case for LayerNormGrad + AddN
    grad_output_nodes[0] = grad_mul_node_1;
    GetOutputRelations(grad_output_relations, grad_output_nodes);
    // useless nodes will be removed by GE
    Status grad_ret = turbo_instance.MultiInOne(layer_norm_grad_node, grad_input_relations, grad_output_relations, {});
    FUSION_PASS_CHECK(grad_ret != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to fuse LayerNormGrad."), return FAILED);
    FUSION_PASS_CHECK(UpdateAddN(turbo_instance, layer_norm_grad_node, add_n_node_1, input_size) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to update add_n1."), return FAILED);
  } else {
    GetOutputRelations(grad_output_relations, grad_output_nodes);
    Status grad_ret = turbo_instance.MultiInOne(layer_norm_grad_node, grad_input_relations, grad_output_relations, {});
    FUSION_PASS_CHECK(grad_ret != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to fuse LayerNormGrad."), return FAILED);
  }

  // to be LayerNorm
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_0(squared_difference_node, mean_node_1);
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_1(mul_node_1, rsqrt_node);
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_2(sub_node, mul_node_2);
  std::vector<std::pair<ge::NodePtr, ge::NodePtr>> input_node_pairs = {input_pair_0, input_pair_1, input_pair_2};
  std::vector<ge::NodePtr> output_nodes = {add_node_2, mean_node_1, rsqrt_node};
  Relations input_relations;
  Relations output_relations;
  GetInputRelations(input_relations, input_node_pairs);
  GetOutputRelations(output_relations, output_nodes);
  Status ret = turbo_instance.MultiInOne(layer_norm_node, input_relations, output_relations, {});
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to fuse LayerNorm."), return FAILED);

  new_nodes.emplace_back(layer_norm_node);
  new_nodes.emplace_back(layer_norm_grad_node);

  return SUCCESS;
}
REGISTER_PASS("LayerNormTrainingFusionPass", BUILT_IN_GRAPH_PASS, LayerNormTrainingFusionPass);
}  // namespace fe
