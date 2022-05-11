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
 * \file attention_qkv_gradw_fusion_pass.h
 * \brief attention_qkv_gradw_fusion_pass
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_QKV_GRADW_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_QKV_GRADW_FUSION_PASS_H

#include <string>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AttentionQKVGradWFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  bool IsMatch(const ge::NodePtr &conf_trans_node, std::vector<ge::NodePtr> &conf_trans_list,
               std::vector<ge::NodePtr> &matmul_list, std::vector<ge::NodePtr> &reduce_sum_list) const;
  Status ReplaceAttentionQKVGradW(ge::ComputeGraph &graph,
                                  const std::vector<ge::NodePtr> &conf_trans_list,
                                  const std::vector<ge::NodePtr> &matmul_list,
                                  const std::vector<ge::NodePtr> &reduce_sum_list,
                                  ge::NodePtr &new_node);
  const string FUSED_OP_TYPE = "AttentionQKVGradW";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_QKV_GRADW_FUSION_PASS_H
