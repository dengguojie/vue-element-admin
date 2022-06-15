/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file matmul_generalized_ub_fusion.cpp
 * \brief matmul fusion pattern
 */
#include <iostream>
#include <string>

#include "../../../../graph_fusion/ai_core/tbe_ops_pass_util.h"
#include "anchor_util.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pre_trans_func.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "lx_fusion_func.h"
#include "matmul_generalized_ub_fusion.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace std;

namespace fe {
namespace {
static const char kDescTransDataBefore[] = "transdata_before";
static const char kDescMatMul[] = "matmul";
// pattern: dequant/requant/fixpipe
static const char kDescAdjacentMatMul[] = "adjacent_matmul";
static const char kDescElemwise0[] = "elemwise_0";
static const char kDescElemwise1[] = "elemwise_1";
// pattern: quant/CommReduce/confusiontranspose
static const char kDescTermination[] = "termination";
static const char kOpTypeTransData[] = "TransData";
static const char kOpTypeMatMulV2[] = "MatMulV2";
static const char kFusionName[] = "MatMulGeneralizedUbFusion";
}  // namespace

vector<BufferFusionPattern *> MatMulGeneralizedUbFusion::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string name_pass = "MatMulGeneralizedUbFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(name_pass);

  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGW(kFusionName, "new an object failed."), return patterns);
  OP_LOGD(kFusionName, "Start to define %s pass pattern.", name_pass.c_str());
  pattern
      ->AddOpDesc(kDescTransDataBefore, {TBE_PATTERN_OP_TYPE_ANY}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kDescMatMul, {OP_PATTERN_MATMUL, OP_PATTERN_BATCH_MATMUL, OP_PATTERN_GEMM}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kDescAdjacentMatMul, {OP_PATTERN_DEQUANT, OP_PATTERN_REQUANT, OP_PATTERN_FIXPIPE},
                 TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kDescElemwise0, {OP_PATTERN_ELEMWISE, OP_PATTERN_DROPOUTDOMASKV3D, OP_PATTERN_STRIDED_WRITE},
                 TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_MAX, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kDescElemwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_MAX,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kDescTermination, {OP_PATTERN_COMMONREDUCE, OP_PATTERN_QUANT, OP_PATTERN_CONFUSION_TRANSPOSE},
                 TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kDescTransDataBefore, kDescMatMul})
      .SetOutputs(kDescTransDataBefore, {kDescMatMul})
      .SetOutputs(kDescMatMul, {kDescAdjacentMatMul}, TBE_OUTPUT_BRANCH_MULTI, true, true)
      .SetOutputs(kDescAdjacentMatMul, {kDescElemwise0, kDescElemwise1}, TBE_OUTPUT_BRANCH_DEFAULT, true, true)
      .SetOutputs(kDescElemwise0, {kDescTermination}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(kDescTermination, {}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(kDescElemwise1, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern);
  OP_LOGD(kFusionName, "End to define %s pass pattern.", name_pass.c_str());
  return patterns;
}

bool MatMulGeneralizedUbFusion::MatchType(const vector<string> &types, const vector<ge::NodePtr> &nodes_ptr) const {
  if (types.empty() && nodes_ptr.empty()) {
    return true;
  }

  FUSION_PASS_CHECK(nodes_ptr.size() != 1, OP_LOGD(kFusionName, "size of matched node must be 1"), return false);
  FUSION_PASS_CHECK(nodes_ptr[0] == nullptr, OP_LOGD(kFusionName, "pointer of node is null"), return false);
  return find(types.begin(), types.end(), nodes_ptr[0]->GetType()) != types.end();
}

bool MatMulGeneralizedUbFusion::MatchType(const vector<vector<string>> &types,
                                          const vector<ge::NodePtr> &nodes_ptr) const {
  if (types.size() > nodes_ptr.size()) {
    OP_LOGD(kFusionName, "match pattern exceed nodes_ptr");
    return false;
  }

  for (size_t i = 0; i < types.size(); ++i) {
    FUSION_PASS_CHECK(nodes_ptr[i] == nullptr, OP_LOGD(kFusionName, "pointer of node is null"), return false);
    const auto &type = types[i];
    if (find(type.begin(), type.end(), nodes_ptr[i]->GetType()) == type.end()) {
      return false;
    }
  }

  return true;
}

bool MatMulGeneralizedUbFusion::MatchTransDataBefore(const vector<string> &types,
                                                     const vector<ge::NodePtr> &nodes_ptr) const {
  // don't care about matching nodes
  if (types.empty()) {
    return true;
  }

  if (types.size() != 1 || nodes_ptr.size() != 1 || types.front() != kOpTypeTransData) {
    OP_LOGD(kFusionName, "Inner logic error");
    return false;
  }

  return true;
}

void MatMulGeneralizedUbFusion::ConstructFusionNodes(const BufferFusionMapping &mapping,
                                                     vector<ge::NodePtr> &fusion_nodes) const {
  for (const auto &item : mapping) {
    if (item.first->desc_name == kDescTransDataBefore) {
      continue;
    }

    const auto &ptr_nodes = item.second;
    for (const auto &ptr_node : ptr_nodes) {
      fusion_nodes.emplace_back(ptr_node);
    }
  }
}

size_t MatMulGeneralizedUbFusion::GetRealIdx(size_t ori_idx, const struct OffsetIndex &offset_index) {
  auto offset = offset_index.offset;

  for (size_t i = 0; i < offset_index.ignore_input_indices.size(); ++i) {
    auto ignore_idx = offset_index.ignore_input_indices[i];
    if (ori_idx == ignore_idx) {
      return SIZE_MAX;
    }

    if (ori_idx < ignore_idx) {
      break;
    }

    --offset;
  }

  return ori_idx + offset;
}

void MatMulGeneralizedUbFusion::TraverseMaps2(const AxisSplitMap &map1, const OutputSplitInfoPtr &output_ptr_map1,
                                              const std::vector<AxisSplitMap> &maps2,
                                              const struct OffsetIndex &offset_index,
                                              vector<AxisSplitMap> &intersect_maps) {
  auto index_output_map1 = output_ptr_map1->GetIndex();
  auto axis_output_map1 = output_ptr_map1->GetAxis();

  for (const auto &map2 : maps2) {
    auto inputs_map2 = map2.GetInputSplitInfoVec();
    auto outputs_ptr_map2 = map2.GetOutputSplitInfos();
    if (inputs_map2.empty() || outputs_ptr_map2.empty()) {
      continue;
    }

    for (const auto &output_ptr_map2 : outputs_ptr_map2) {
      if (output_ptr_map2 == nullptr) {
        continue;
      }

      if (output_ptr_map2->GetIndex() != index_output_map1) {
        continue;
      }
      if (output_ptr_map2->GetAxis() != axis_output_map1) {
        continue;
      }

      intersect_maps.emplace_back(GenFusionSplitMap(map1, inputs_map2, offset_index));
    }
  }
}

AxisSplitMap MatMulGeneralizedUbFusion::GenFusionSplitMap(const AxisSplitMap &map1,
                                                          const vector<InputSplitInfo> &inputs_map2,
                                                          const struct OffsetIndex &offset_index) {
  AxisSplitMap intersect_map{map1};

  for (auto input_map2 : inputs_map2) {
    auto real_idx = GetRealIdx(input_map2.GetIndex(), offset_index);
    if (real_idx == SIZE_MAX) {
      continue;
    }

    input_map2.SetIndex(real_idx);
    intersect_map.AddInputSplitInfo(input_map2);
  }
  return intersect_map;
}

std::vector<AxisSplitMap> MatMulGeneralizedUbFusion::IntersectSplitMap(const std::vector<AxisSplitMap> &maps1,
                                                                       const std::vector<AxisSplitMap> &maps2,
                                                                       const struct OffsetIndex &offset_index) {
  vector<AxisSplitMap> intersect_maps;

  for (const auto &map1 : maps1) {
    auto outputs_ptr_map1 = map1.GetOutputSplitInfos();
    if (outputs_ptr_map1.empty()) {
      continue;
    }

    for (const auto &output_ptr_map1 : outputs_ptr_map1) {
      if (output_ptr_map1 == nullptr) {
        continue;
      }

      TraverseMaps2(map1, output_ptr_map1, maps2, offset_index, intersect_maps);
    }
  }
  return intersect_maps;
}

std::vector<uint32_t> MatMulGeneralizedUbFusion::GetIgnoreInputIndices(const ge::NodePtr &node_ptr_curr,
                                                                       const std::vector<ge::NodePtr> &fusion_nodes) {
  vector<uint32_t> ignore_input_indices;
  uint32_t size = node_ptr_curr->GetAllInDataAnchorsSize();

  for (uint32_t idx = 0; idx < size; ++idx) {
    auto in_data_anchor = node_ptr_curr->GetInDataAnchor(idx);
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }

    for (auto node_ptr_prev : fusion_nodes) {
      if (out_data_anchor->GetOwnerNode() == node_ptr_prev) {
        ignore_input_indices.push_back(idx);
      }
    }
  }

  sort(ignore_input_indices.begin(), ignore_input_indices.end());
  return ignore_input_indices;
}

bool MatMulGeneralizedUbFusion::IntersectSplitMapWithElemwise(ge::NodePtr &node,
                                                              const vector<AxisSplitMap> &split_maps_prev,
                                                              vector<AxisSplitMap> *ptr_split_maps_intersect,
                                                              size_t *index_already_provide_split_info,
                                                              const std::vector<ge::NodePtr> &fusion_nodes) {
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  vector<AxisSplitMap> curr_split_maps;

  if (!GetSplitMap(curr_split_maps, node, node->GetName(), fusion_type, min_tbe_l1space)) {
    return false;
  }
  auto ignore_input_indices = GetIgnoreInputIndices(node, fusion_nodes);
  struct OffsetIndex offset_index {
    .offset = *index_already_provide_split_info, .ignore_input_indices = ignore_input_indices
  };
  *ptr_split_maps_intersect = IntersectSplitMap(split_maps_prev, curr_split_maps, offset_index);
  *index_already_provide_split_info += node->GetAllInDataAnchorsSize() - ignore_input_indices.size();
  return true;
}

void MatMulGeneralizedUbFusion::SetSplitInfo(const BufferFusionMapping &mapping,
                                             std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> nodes_transdata_before = GetMatchedNodesByDescName(kDescTransDataBefore, mapping);
  vector<ge::NodePtr> nodes_mm = GetMatchedNodesByDescName(kDescMatMul, mapping);
  vector<ge::NodePtr> nodes_adjacent_mm = GetMatchedNodesByDescName(kDescAdjacentMatMul, mapping);
  vector<ge::NodePtr> nodes_elemwise_0 = GetMatchedNodesByDescName(kDescElemwise0, mapping);
  vector<ge::NodePtr> nodes_elemwise_1 = GetMatchedNodesByDescName(kDescElemwise1, mapping);
  vector<ge::NodePtr> nodes_termination = GetMatchedNodesByDescName(kDescTermination, mapping);
  vector<vector<ge::NodePtr>> ptr_nodes = {nodes_adjacent_mm, nodes_elemwise_0, nodes_elemwise_1, nodes_termination};

  bool is_not_support = ((!nodes_transdata_before.empty() && nodes_transdata_before[0]->GetType() == "TransData") ||
                         (!nodes_adjacent_mm.empty() && nodes_adjacent_mm[0]->GetType() == "FixPipe") ||
                         (!nodes_termination.empty() && nodes_termination[0]->GetType() == "ConfusionTransposeD"));
  if (is_not_support) {
    OP_LOGW(kFusionName, "split not support this op type");
    return;
  }

  vector<AxisSplitMap> split_maps_prev;
  vector<AxisSplitMap> split_maps_fusion_op;
  vector<AxisSplitMap> *ptr_split_maps_prev;
  vector<AxisSplitMap> *ptr_split_maps_fusion_op;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;

  if (!GetSplitMap(split_maps_prev, nodes_mm[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
    return;
  }
  size_t index_already_provide_split_info = nodes_mm[0]->GetInDataNodes().size();

  ptr_split_maps_prev = &split_maps_prev;
  ptr_split_maps_fusion_op = &split_maps_fusion_op;
  // only support elemwise0,elemwise1,termination,adjacent, except transpose and fixpipe and transdata not support
  for (const auto &nodes : ptr_nodes) {
    if (!nodes.empty()) {
      for (auto node : nodes) {
        FUSION_PASS_CHECK(!IntersectSplitMapWithElemwise(node, *ptr_split_maps_prev, ptr_split_maps_fusion_op,
                                                         &index_already_provide_split_info, fusion_nodes),
                          OP_LOGW(kFusionName, "failed to get split maps of %s", node->GetName().c_str()), return );
        ptr_split_maps_prev = &split_maps_fusion_op;
        ptr_split_maps_fusion_op = &split_maps_prev;
      }
    }
  }

  SetSplitMap(*ptr_split_maps_fusion_op, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

bool MatMulGeneralizedUbFusion::CheckMatchType(const BufferFusionMapping &mapping, const MatchPattern &pattern) const {
  vector<ge::NodePtr> nodes_transdata_before = GetMatchedNodesByDescName(kDescTransDataBefore, mapping);
  vector<ge::NodePtr> nodes_mm = GetMatchedNodesByDescName(kDescMatMul, mapping);
  vector<ge::NodePtr> nodes_adjacent_mm = GetMatchedNodesByDescName(kDescAdjacentMatMul, mapping);
  vector<ge::NodePtr> nodes_elemwise_0 = GetMatchedNodesByDescName(kDescElemwise0, mapping);
  vector<ge::NodePtr> nodes_elemwise_1 = GetMatchedNodesByDescName(kDescElemwise1, mapping);
  vector<ge::NodePtr> nodes_termination = GetMatchedNodesByDescName(kDescTermination, mapping);

  if (!MatchTransDataBefore(pattern.type_transdata_before, nodes_transdata_before)) {
    OP_LOGD(kFusionName, "not match transdata before");
    return false;
  }
  if (!MatchType(pattern.type_mm, nodes_mm)) {
    OP_LOGD(kFusionName, "not match mm");
    return false;
  }
  if (!MatchType(pattern.type_adjacent_mm, nodes_adjacent_mm)) {
    OP_LOGD(kFusionName, "not match adjacent");
    return false;
  }
  if (!MatchType(pattern.type_elemwise_0, nodes_elemwise_0)) {
    OP_LOGD(kFusionName, "not match elemwise_0");
    return false;
  }
  if (!MatchType(pattern.type_elemwise_1, nodes_elemwise_1)) {
    OP_LOGD(kFusionName, "not match elemwise_1");
    return false;
  }
  if (!MatchType(pattern.type_termination, nodes_termination)) {
    OP_LOGD(kFusionName, "not match termination");
    return false;
  }

  OP_LOGD(kFusionName, "match type successfully!");
  return true;
}

bool MatMulGeneralizedUbFusion::CheckShapeType(bool has_unknown_shape, const MatchPattern &pattern) const {
  if (has_unknown_shape && pattern.type_shape == ONLY_SUPPORT_STATIC) {
    OP_LOGD(kFusionName, "only support static shape, but graph is dynamic shape");
    return false;
  }
  if (!has_unknown_shape && pattern.type_shape == ONLY_SUPPORT_DYNAMIC) {
    OP_LOGD(kFusionName, "only support dynamic shape, but graph is static shape");
    return false;
  }

  return true;
}

Status MatMulGeneralizedUbFusion::HelpGetFusionNodes(const BufferFusionMapping &mapping,
                                                     vector<ge::NodePtr> &fusion_nodes,
                                                     const vector<MatchPattern> &match_patterns) {
  OP_LOGD(kFusionName, "Begin to do MatMulGeneralizedUbFusion!");
  // step1: calculate real shape_type
  bool has_unknown_shape = false;
  for (const auto &fusion_map : mapping) {
    for (const auto &node_ptr : fusion_map.second) {
      has_unknown_shape = has_unknown_shape || HasUnKnowShape(node_ptr);
      if (has_unknown_shape) {
        break;
      }
    }
    if (has_unknown_shape) {
      break;
    }
  }

  OP_LOGD(kFusionName, "real shape_type is %s", has_unknown_shape ? "dynamic" : "static");
  for (const auto &pattern : match_patterns) {
    // avoid changes to fusion_nodes from the last loop
    fusion_nodes.clear();

    // step1: check type of shape
    if (!CheckShapeType(has_unknown_shape, pattern)) {
      continue;
    }

    // step2: match type
    if (!CheckMatchType(mapping, pattern)) {
      continue;
    }

    // step3: special check(process the pre-fused TransData and put it into fusion_nodes)
    if (pattern.is_matched != nullptr && !pattern.is_matched(mapping, &fusion_nodes)) {
      continue;
    }
    OP_LOGD(kFusionName, "match special restriction");

    // step4: construct fusion nodes
    ConstructFusionNodes(mapping, fusion_nodes);
    OP_LOGD(kFusionName, "construct fusion nodes");
    break;
  }

  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(kFusionName, "End MatMulGeneralizedUbFusion!");
  return SUCCESS;
}
}  // namespace fe
