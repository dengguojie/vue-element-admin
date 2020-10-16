/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief logsoftmaxgrad fusion pass
 *
 */

#ifndef FE_FUSION_LOGSOFTMAXGRAD_FUSION_PASS_H
#define FE_FUSION_LOGSOFTMAXGRAD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LogSoftmaxGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    Status IsMatch(ge::NodePtr sumNode, ge::NodePtr subNode,
                          ge::NodePtr expNode, ge::NodePtr mulNode);
    Status DoFusion(ge::ComputeGraph &graph, ge::NodePtr sumNode,
                           ge::NodePtr subNode, ge::NodePtr expNode,
                           ge::NodePtr mulNode, vector<ge::NodePtr> &fusionNodes);
    Status UpdateAttr(ge::NodePtr sumNode, ge::NodePtr subNode);
    Status LinkOutputEdge(ge::NodePtr oldNode, ge::NodePtr newNode);
    const string FUSED_OP_TYPE = "LogSoftmaxGrad";
};
}  // namespace fe
#endif  // FE_FUSION_LOGSOFTMAXGRAD_FUSION_PASS_H_
