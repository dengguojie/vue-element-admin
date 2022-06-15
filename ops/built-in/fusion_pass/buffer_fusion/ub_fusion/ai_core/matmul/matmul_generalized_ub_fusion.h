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
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_GENERALIZED_UB_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_GENERALIZED_UB_FUSION_H_

#include <vector>

#include "common/lxfusion_json_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
const vector<string> type_matmul = {"BatchMatMulV2", "BatchMatMul", "MatMulV2", "MatMul"};
using func_ptr_is_matched = bool (*)(const fe::BufferFusionMapping &mapping, vector<ge::NodePtr> *fusion_nodes);

struct OffsetIndex {
  size_t offset;
  std::vector<uint32_t> ignore_input_indices;
};

class MatchPattern {
 public:
  MatchPattern() {}

  MatchPattern &SetTransDataBefore(const vector<string> &val) {
    type_transdata_before = val;
    return *this;
  }

  MatchPattern &SetMM(const vector<string> &val) {
    type_mm = val;
    return *this;
  }

  MatchPattern &SetAdjacentMM(const vector<string> &val) {
    type_adjacent_mm = val;
    return *this;
  }

  MatchPattern &SetElemwise0(const vector<vector<string>> &val) {
    type_elemwise_0 = val;
    return *this;
  }

  MatchPattern &SetElemwise1(const vector<vector<string>> &val) {
    type_elemwise_1 = val;
    return *this;
  }

  MatchPattern &SetTermination(const vector<string> &val) {
    type_termination = val;
    return *this;
  }

  MatchPattern &SetTypeShape(const fe::ShapeTypeRule &val) {
    type_shape = val;
    return *this;
  }

  MatchPattern &SetIsMatched(func_ptr_is_matched val) {
    is_matched = val;
    return *this;
  }

  vector<string> type_transdata_before;
  vector<string> type_mm;
  vector<string> type_adjacent_mm;
  vector<vector<string>> type_elemwise_0;
  vector<vector<string>> type_elemwise_1;
  vector<string> type_termination;
  fe::ShapeTypeRule type_shape = ONLY_SUPPORT_STATIC;
  bool (*is_matched)(const fe::BufferFusionMapping &mapping, vector<ge::NodePtr> *fusion_nodes) = nullptr;
};

class MatMulGeneralizedUbFusion : public BufferFusionPassBase {
 public:
  explicit MatMulGeneralizedUbFusion() {}
  ~MatMulGeneralizedUbFusion() {}
  vector<BufferFusionPattern *> DefinePatterns() override;
  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status HelpGetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes,
                            const vector<MatchPattern> &match_patterns);
  bool MatchType(const vector<string> &types, const vector<ge::NodePtr> &nodes_ptr) const;
  bool MatchType(const vector<vector<string>> &types, const vector<ge::NodePtr> &nodes_ptr) const;
  bool MatchTransDataBefore(const vector<string> &types, const vector<ge::NodePtr> &nodes_ptr) const;
  void ConstructFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) const;
  /*
   * @brief: only support elemwise0,elemwise1,termination,adjacent,
             transpose and fixpipe and transdata not support,
             other scenes need to add by yourself.
   * @param [in] mapping: nodes matched by pattern
   * @param [out] fusion_nodes: fusioned nodes
   * @return void.
   */
  virtual void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  bool CheckMatchType(const BufferFusionMapping &mapping, const MatchPattern &pattern) const;
  bool CheckShapeType(bool has_unknown_shape, const MatchPattern &pattern) const;

 private:
  const string FUSED_OP_TYPE = "FusedOp";
  static std::vector<uint32_t> GetIgnoreInputIndices(const ge::NodePtr &node_ptr_curr,
                                                     const std::vector<ge::NodePtr> &fusion_nodes);
  static std::vector<AxisSplitMap> IntersectSplitMap(const std::vector<AxisSplitMap> &map1,
                                                     const std::vector<AxisSplitMap> &map2,
                                                     const struct OffsetIndex &offset_index);
  static size_t GetRealIdx(size_t ori_idx, const struct OffsetIndex &offset_index);
  static bool IntersectSplitMapWithElemwise(ge::NodePtr &nodes, const vector<AxisSplitMap> &split_maps_prev,
                                            vector<AxisSplitMap> *ptr_split_maps_intersect,
                                            size_t *index_already_provide_split_info,
                                            const std::vector<ge::NodePtr> &fusion_nodes);
  static AxisSplitMap GenFusionSplitMap(const AxisSplitMap &map1, const vector<InputSplitInfo> &inputs_map2,
                                        const struct OffsetIndex &offset_index);
  static void TraverseMaps2(const AxisSplitMap &map1, const OutputSplitInfoPtr &output_ptr_map1,
                            const std::vector<AxisSplitMap> &maps2, const struct OffsetIndex &offset_index,
                            vector<AxisSplitMap> &intersect_maps);
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_GENERALIZED_UB_FUSION_H_
