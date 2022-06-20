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
 * \file swin_attention_score_fusion_pass.h
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_SCORE_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_SCORE_FUSION_PASS_H
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SwinAttentionScoreFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

private:
  ge::NodePtr batch_matmul_node1 = nullptr;
  ge::NodePtr batch_matmul_node2 = nullptr;
  ge::NodePtr batch_matmul_node3 = nullptr;
  ge::NodePtr softmax_node = nullptr;

  Status AddOutputEdgeForNode(ge::NodePtr ori_node, ge::NodePtr new_node, int unlinkIndex, int new_node_index) const;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_SCORE_FUSION_PASS_H