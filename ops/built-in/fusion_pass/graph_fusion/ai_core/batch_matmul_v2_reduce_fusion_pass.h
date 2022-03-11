/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file batch_matmul_v2_reduce_fusion_pass.h
 * \brief batch_matmul_v2_reduce_fusion_pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_REDUCE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_REDUCE_FUSION_PASS_H_

#include <string>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMatMulV2ReduceFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) override;

 private:
  bool CheckProduct(const std::vector<int64_t> &shape, std::size_t len) const;
  // BatchMatMulV2 --> ReduceSumD --> Output
  bool IsMatchScenario1(const ge::NodePtr &fused_node) const;
  // BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output
  bool IsMatchScenario2(const ge::NodePtr &fused_node) const;
  bool CheckNeedChange(const ge::NodePtr &fused_node, const vector<int64_t> &shape_x, const vector<int64_t> &shape_y,
                       const vector<int64_t> &product_shape_x, const vector<int64_t> &product_shape_y) const;
  Status CreateReshapeNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node, const ge::OutDataAnchorPtr &out_anchor,
                           const vector<int64_t> &shape, ge::NodePtr &shape_node) const;
  Status InsertReshapeNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node, int32_t index,
                           const vector<int64_t> &new_shape) const;
  Status InsertTransposeDNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                              std::tuple<int, std::vector<int64_t>, std::vector<int32_t>> &param,
                              ge::NodePtr &transposedNode) const;
  Status LinkEdge(ge::NodePtr &fused_node, ge::NodePtr &tgt_node) const;
  Status DealWithInputWithKOne(
      ge::ComputeGraph &graph, ge::NodePtr &fused_node,
      std::tuple<int, std::vector<int64_t>, int, std::vector<int64_t>, std::vector<bool>> &param) const;
  Status DoFusionWithKOne(ge::ComputeGraph &graph, ge::NodePtr &fused_node, const vector<int64_t> &new_x1_out_shape,
                          const vector<int64_t> &new_x2_out_shape, const vector<bool> &trans) const;
  Status DealWithInputWithKNotOne(
      ge::ComputeGraph &graph, ge::NodePtr &fused_node,
      std::tuple<int, std::vector<int64_t>, int, std::vector<int64_t>, std::vector<bool>> &param) const;
  Status DoFusionWithKNotOne(ge::ComputeGraph &graph, ge::NodePtr &fused_node, const vector<int64_t> &new_x1_out_shape,
                             const vector<int64_t> &new_x2_out_shape, const vector<bool> &trans) const;
  const string FUSED_OP_TYPE = "BatchMatMulV2";
};
} // namespace fe
#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_REDUCE_FUSION_PASS_H_
