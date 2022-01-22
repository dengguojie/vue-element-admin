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
 * \file matmul_fusion_pass.h
 * \brief matmul reshape fusion (reshape-matmul--reshape)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BAT_MATMUL_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BAT_MATMUL_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMatMulFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

private:
  bool CheckIsNeedFusion(const ge::NodePtr& fused_node) const;
  bool CheckAndDoTransposeFusion(ge::ComputeGraph& graph, const ge::NodePtr& fused_node) const;
  bool CheckTransposeFusion(const ge::NodePtr& transpose_node) const;
  Status DoTransposeFusion(const ge::NodePtr& transpose_node, const ge::NodePtr& fused_node, int data_index,
                           const string& attr_name) const;
  Status LinkEdge(const ge::NodePtr& transpose_node, const ge::NodePtr& fused_node, int data_index) const;
  Status CreateMatMulNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node, ge::NodePtr& new_node) const;
  Status AddEdgeForMatMulNode(const ge::NodePtr& fused_node, const ge::NodePtr& matmul_node) const;
  Status RemoveFusedNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node) const;
  const string FUSED_OP_TYPE = "BatchMatMulFusionPass";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BAT_MATMUL_FUSION_PASS_H_
