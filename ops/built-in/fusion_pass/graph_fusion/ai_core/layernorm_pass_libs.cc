/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file layernorm_pass_libs.cc
 * \brief used for layernorm & layernormgrad fusion
 */
#include "layernorm_pass_libs.h"
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
Status CheckNullPtr(const std::vector<ge::NodePtr>& pattern_nodes) {
  for (const auto& node_ptr : pattern_nodes) {
    if (node_ptr == nullptr || node_ptr->GetOpDesc() == nullptr) {
      return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

Status GetReduceOpAttr(std::vector<int64_t>& axes, bool& keep_dims, const ge::NodePtr& node) {
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

Status CheckReduceOpAttr(const std::string& fused_op_type, const std::vector<int64_t>& axes_1, const bool keep_dims_1,
                         const ge::NodePtr& mean_node_1, const ge::NodePtr& mean_node_2) {
  std::vector<int64_t> axes_2 = {-1};
  bool keep_dims_2 = false;
  FUSION_PASS_CHECK(GetReduceOpAttr(axes_2, keep_dims_2, mean_node_2) != SUCCESS,
                    OP_LOGD(fused_op_type, "failed to get attributes of mean2."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!(axes_1.size() == 1 && axes_1 == axes_2),
                    OP_LOGD(fused_op_type, "the axes of mean nodes are not same."), return NOT_CHANGED);

  std::vector<bool> keep_dims = {keep_dims_1, keep_dims_2};
  FUSION_PASS_CHECK(std::find(keep_dims.begin(), keep_dims.end(), false) != keep_dims.end(),
                    OP_LOGD(fused_op_type, "the keep_dims of mean is false."), return NOT_CHANGED);

  auto input_desc = mean_node_1->GetOpDesc()->MutableInputDesc(0);
  FUSION_PASS_CHECK(input_desc == nullptr, OP_LOGD(fused_op_type, "mean1's input is nullptr."), return NOT_CHANGED);
  size_t dims_size = input_desc->MutableShape().GetDims().size();
  FUSION_PASS_CHECK(dims_size < 1,
                    OP_LOGD(fused_op_type, "input shape should be greater than 0."), return NOT_CHANGED);
  FUSION_PASS_CHECK(axes_1[0] != -1 && axes_1[0] != static_cast<int64_t>(dims_size - 1),
                    OP_LOGD(fused_op_type, "the axes is not the last axis."), return NOT_CHANGED);

  return SUCCESS;
}

Status SetLayerNormAttr(const std::string& fused_op_type, const ge::NodePtr& node,
                        const std::vector<int64_t>& axes_vec, const ge::NodePtr& add_1) {
  auto add_op = ge::OpDescUtils::CreateOperatorFromNode(add_1);
  std::vector<float> epsilon_vec = {1e-7};
  if (!ops::GetConstIntData(add_op, 0, epsilon_vec)) {
    if (!ops::GetConstIntData(add_op, 1, epsilon_vec)) {
      OP_LOGD(fused_op_type, "set epsilon to default value.");
    }
  }

  auto node_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(node_desc == nullptr, OP_LOGD(fused_op_type, "OpDesc is null, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(node_desc, "begin_norm_axis", axes_vec[0]),
                    OP_LOGD(fused_op_type, "failed to set begin_norm_axis, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(node_desc, "begin_params_axis", -1),
                    OP_LOGD(fused_op_type, "failed to set begin_params_axis, do nothing."), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(node_desc, "epsilon", epsilon_vec[0]),
                    OP_LOGD(fused_op_type, "failed to set epsilon, do nothing."), return NOT_CHANGED);

  return SUCCESS;
}

void GetInputRelations(Relations& input_relations, const std::vector<std::pair<ge::NodePtr, ge::NodePtr>>& pairs) {
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

void GetOutputRelations(Relations& output_relations, const std::vector<ge::NodePtr>& nodes) {
  for (size_t idx = 0; idx < nodes.size(); idx++) {
    NodeIndex node_index(nodes[idx], 0, PEER);
    std::vector<NodeIndex> node_index_vec = {node_index};
    output_relations.Add(static_cast<int32_t>(idx), node_index_vec);
  }
}
}  // namespace fe
