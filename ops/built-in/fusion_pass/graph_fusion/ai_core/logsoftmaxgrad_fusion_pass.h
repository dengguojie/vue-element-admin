/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file logsoftmaxgrad_fusion_pass.h
 * \brief logsoftmaxgrad fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LOGSOFTMAXGRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LOGSOFTMAXGRAD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LogSoftmaxGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status IsMatch(ge::NodePtr sumNode, ge::NodePtr subNode, ge::NodePtr expNode, ge::NodePtr mulNode);
  Status DoFusion(ge::ComputeGraph& graph, ge::NodePtr sumNode, ge::NodePtr subNode, ge::NodePtr expNode,
                  ge::NodePtr mulNode, vector<ge::NodePtr>& fusionNodes);
  Status UpdateAttr(ge::NodePtr sumNode, ge::NodePtr subNode) const;
  Status LinkOutputEdge(ge::NodePtr oldNode, ge::NodePtr newNode);
  const string FUSED_OP_TYPE = "LogSoftmaxGrad";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LOGSOFTMAXGRAD_FUSION_PASS_H_
