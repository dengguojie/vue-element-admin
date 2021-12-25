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
 * \file nll_loss_fusion_pass.h
 * \brief NLLLoss fusion pass(NLLLoss --> NLLLoss(sum) & reduce(mean))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_NLLLOSS_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_NLLLOSS_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class NLLLossFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  ge::NodePtr AddNLLLossSumNode(ge::NodePtr nll_loss_node, ge::ComputeGraph& graph,
                                vector<ge::NodePtr>& new_nodes, bool& fail_status);
  ge::NodePtr AddDivNode(ge::NodePtr nll_loss_node, ge::NodePtr nll_loss_sum_node, ge::ComputeGraph& graph,
                         vector<ge::NodePtr>& new_nodes, bool& fail_status);
  bool IsFusionPassEnable(ge::NodePtr nll_loss_node, string reduction, string aic_version, bool is_unknown_shape);
  const string FUSED_OP_TYPE = "NLLLoss";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_NLLLOSS_FUSION_PASS_H_
