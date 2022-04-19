/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file batch_matmul_elementwise_ub_fusion.cpp
 * \brief tbe batchmatmul and all elementwise ops fusion pattern
 */
#include "batch_matmul_elementwise_ub_fusion.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "lx_fusion_func.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
namespace {
static const char PATTERN_BATCH_MATMUL[] = "batchmatmul";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_ELEM_1[] = "elemwise1";
static const char PATTERN_ELEM_2[] = "elemwise2";
static vector<string> elem_typelist = {"FusedMulAdd", "Add", "Div", "RealDiv", "Relu", "ReluGrad"};
static vector<string> elem1_typelist = {"Add", "Relu", "FusedMulAdd"};
static const char PATTERN_OTHER_INPUT[] = "InputData";
static const char PATTERN_OTHER_INPUT1[] = "InputData1";
static const char PATTERN_OTHER_OUTPUT[] = "OutputData";
}  // namespace

/*
 * @brief: define Matmul and element-wise op fusion pattern
 *
 * Matmul + Elemwise + (Elemwise1)
 *
 * fusion node: Matmul, Elemwise, (Elemwise1)
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeBatchMatmulElementWiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName = "TbeBatchMatmulElemElemPattern";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM, {PATTERN_ELEM_1}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM_1, {}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  string passName1 = "TbeBatchMatmulElemPattern";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM, {}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());

  /*
   * BatchmatmulV2 --> Mul --> Sigmoid --> Mul --> Output
   *       \_______________________________/
   */
  string passName2 = "TbeBatchMatmulMulSigmoidMulPattern";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM_1, PATTERN_OTHER_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELEM_1})
      .SetOutputs(PATTERN_ELEM_1, {PATTERN_ELEM_2}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(PATTERN_ELEM_2, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());

  return patterns;
}

size_t TbeBatchMatmulElementWiseFusionPass::GetRealIdx(size_t ori_idx, const struct OffsetIndex &offset_index) {
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

void TbeBatchMatmulElementWiseFusionPass::TraverseMaps2(const AxisSplitMap &map1,
                                                        const OutputSplitInfoPtr output_ptr_map1,
                                                        const std::vector<AxisSplitMap> &maps2,
                                                        const struct OffsetIndex &offset_index,
                                                        vector<AxisSplitMap> *intersect_maps) {
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

      intersect_maps->emplace_back(GenFusionSplitMap(map1, inputs_map2, offset_index));
    }
  }
}

AxisSplitMap TbeBatchMatmulElementWiseFusionPass::GenFusionSplitMap(const AxisSplitMap &map1,
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

std::vector<AxisSplitMap> TbeBatchMatmulElementWiseFusionPass::IntersectSplitMap(
    const std::vector<AxisSplitMap> &maps1, const std::vector<AxisSplitMap> &maps2,
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

      TraverseMaps2(map1, output_ptr_map1, maps2, offset_index, &intersect_maps);
    }
  }
  return intersect_maps;
}

std::vector<uint32_t> TbeBatchMatmulElementWiseFusionPass::GetIgnoreInputIndices(
    const ge::NodePtr &node_ptr_curr, const std::vector<ge::NodePtr> &fusion_nodes) {
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

bool TbeBatchMatmulElementWiseFusionPass::IntersectSplitMapWithElemwise(ge::NodePtr &node,
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

void TbeBatchMatmulElementWiseFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                       std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemWiseNodes1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);
  vector<ge::NodePtr> elemWiseNodes2 = GetMatchedNodesByDescName(PATTERN_ELEM_2, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (elemWiseNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Elemwise node not matched");
    return;
  }
  FUSION_PASS_CHECK(matmulNodes[0]->GetInDataNodes().size() <= 0,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "matmulNodes's input can not <= 0."), return);
  vector<AxisSplitMap> split_maps_prev;
  vector<AxisSplitMap> split_maps_fusion_op;
  vector<AxisSplitMap>* ptr_split_maps_prev;
  vector<AxisSplitMap>* ptr_split_maps_fusion_op;

  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!GetSplitMap(split_maps_prev, matmulNodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
    return;
  }
  size_t index_already_provide_split_info = matmulNodes[0]->GetInDataNodes().size();

  ptr_split_maps_prev = &split_maps_prev;
  ptr_split_maps_fusion_op = &split_maps_fusion_op;
  FUSION_PASS_CHECK(
      !IntersectSplitMapWithElemwise(elemWiseNodes[0], *ptr_split_maps_prev, ptr_split_maps_fusion_op,
                                     &index_already_provide_split_info, fusion_nodes),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to get split maps of %s", elemWiseNodes[0]->GetName().c_str()), return);

  if (!elemWiseNodes1.empty()) {
    ptr_split_maps_prev = &split_maps_fusion_op;
    ptr_split_maps_fusion_op = &split_maps_prev;
    FUSION_PASS_CHECK(
        !IntersectSplitMapWithElemwise(elemWiseNodes1[0], *ptr_split_maps_prev, ptr_split_maps_fusion_op,
                                       &index_already_provide_split_info, fusion_nodes),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to get split maps of %s", elemWiseNodes1[0]->GetName().c_str()),
        return);
  }

  if (!elemWiseNodes2.empty()) {
    ptr_split_maps_prev = &split_maps_fusion_op;
    ptr_split_maps_fusion_op = &split_maps_prev;
    FUSION_PASS_CHECK(
        !IntersectSplitMapWithElemwise(elemWiseNodes2[0], *ptr_split_maps_prev, ptr_split_maps_fusion_op,
                                       &index_already_provide_split_info, fusion_nodes),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to get split maps of %s", elemWiseNodes2[0]->GetName().c_str()),
        return);
  }

  SetSplitMap(*ptr_split_maps_fusion_op, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

Status TbeBatchMatmulElementWiseFusionPass::CheckPattern1(const BufferFusionMapping &mapping) const {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemNode1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);

  FUSION_PASS_CHECK(elemNode.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return FAILED);
  auto ret = find(elem_typelist.begin(), elem_typelist.end(), elemNode[0]->GetType());
  if (ret == elem_typelist.end()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add, div, muladd and Relu in first elemwise");
    return FAILED;
  }

  if (!elemNode1.empty()) {
    ret = find(elem1_typelist.begin(), elem1_typelist.end(), elemNode1[0]->GetType());
    if (ret == elem1_typelist.end()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add, relu and muladd in second elemwise");
      return FAILED;
    }
  }

  return SUCCESS;
}

Status TbeBatchMatmulElementWiseFusionPass::CheckPattern2(const BufferFusionMapping &mapping) const {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemNode1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);
  vector<ge::NodePtr> elemNode2 = GetMatchedNodesByDescName(PATTERN_ELEM_2, mapping);
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  FUSION_PASS_CHECK(elemNode.empty() || elemNode1.empty() || elemNode2.empty() || matmulNodes.empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return FAILED);

  bool is_matched = false;
  if (elemNode[0]->GetType() != "Mul" || elemNode1[0]->GetType() != "Mul" || elemNode2[0]->GetType() != "Sigmoid") {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match, node name [%s], [%s], [%s].",
            elemNode[0]->GetType().c_str(), elemNode1[0]->GetType().c_str(), elemNode2[0]->GetType().c_str());
    return FAILED;
  }

  auto elem_name = elemNode[0]->GetName();
  auto out_nodes = matmulNodes[0]->GetOutDataNodes();
  if (out_nodes.size() != 2) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "matmulNode output size not match!");
    return FAILED;
  }

  for (auto &node : out_nodes) {
    is_matched = is_matched || (node->GetName() == elem_name);
  }
  FUSION_PASS_CHECK(!is_matched, OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node name not match!"), return FAILED);

  return SUCCESS;
}

Status TbeBatchMatmulElementWiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                           vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeBatchMatmulElementWiseFusionPass!");

  if (SUCCESS != CheckPattern1(mapping) && SUCCESS != CheckPattern2(mapping)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "CheckPattern failed!");
    return SUCCESS;
  }

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  for (const auto &matmulNode : matmulNodes) {
    auto input0desc = GetCurrNodeInputDesc(matmulNode, 0);
    auto input1desc = GetCurrNodeInputDesc(matmulNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                      return SUCCESS);
    FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                      return SUCCESS);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> allDim;
    allDim.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDim.begin());
    for (auto singleDim : allDim) {
      if (singleDim < 0) {
        fusion_nodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeBatchMatmulElementWiseFusionPass!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeBatchMatmulElementWiseFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeBatchMatmulElementWiseFusionPass);
} // namespace fe
