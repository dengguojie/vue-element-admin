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
 * \file batch_matmul_v2_reshape_fusion_pass.h
 * \brief batch_matmul_v2_reshape_fusion_pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_RESHAPE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_RESHAPE_FUSION_PASS_H_

#include <string>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMatMulV2ReshapeFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) override;

 private:
  bool CheckProduct(const std::vector<int64_t> &shape, std::size_t len);
  bool CheckNeedChange(const ge::NodePtr &fused_node, const vector<int64_t> &shape_x, const vector<int64_t> &shape_y);
  Status InputInsertReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node, int32_t index,
                                const vector<int64_t> &new_shape);
  Status OutputInsertReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node, int32_t index,
                                 const vector<int64_t> &out_shape);
  Status UpdateOpDescByIndex(const ge::NodePtr &node, const vector<int64_t> &new_shape, int32_t index);
  Status UpdateOpDesc(const ge::NodePtr &node, const vector<int64_t> &new_shape);
  Status ConnectOneElemwise(ge::ComputeGraph &graph, const ge::NodePtr &next_node, const vector<int64_t> &new_shape,
                            const vector<int64_t> &out_shape);
  bool IsMatchScenario1(const ge::NodePtr &fused_node) const; // BatchMatMulV2 --> Add --> Output
  bool IsMatchScenario2(const ge::NodePtr &fused_node) const; // BatchMatMulV2 --> Add --> Add --> Output
  /*
   * BatchMatMulV2 --> Add --> Mul --> Sigmoid --> Mul --> Output
   *                    \__________________________/
   */
  bool IsMatchScenario3(const ge::NodePtr &fused_node) const;
  Status ConnectTwoElemwise(ge::ComputeGraph &graph, const ge::NodePtr &next_node, const vector<int64_t> &new_shape,
                            const vector<int64_t> &out_shape);
  Status ProcessOutNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node, const vector<int64_t> &new_shape,
                        const vector<int64_t> &out_shape);
  Status CreateReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &next_node,
                           const ge::OutDataAnchorPtr &out_anchor, const vector<int64_t> &shape,
                           ge::NodePtr &shape_node);
  const string FUSED_OP_TYPE = "BatchMatMulV2";
};
} // namespace fe
#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_RESHAPE_FUSION_PASS_H_
