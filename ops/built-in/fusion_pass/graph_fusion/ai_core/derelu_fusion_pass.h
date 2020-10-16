/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief derelu fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_DERELU_FUSION_H_
#define _OPTIMIZER_FUSION_DERELU_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"


namespace fe {
class DreluFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
  bool IsUnknownShape(const ge::GeShape &shape);
  Status RemoveNode(ge::NodePtr node, ge::ComputeGraph &graph);
  Status ReplaceNode(ge::NodePtr oldNode, ge::NodePtr newNode, ge::ComputeGraph &graph);
  ge::NodePtr CreateNode(ge::ComputeGraph &graph, ge::NodePtr relu, vector<ge::NodePtr> &fusionNodes);
  const string FUSED_OP_TYPE = "ReluV2_ReluGradV2";
};

}  // namespace fe
#endif  // _OPTIMIZER_FUSION_DERELU_FUSION_H_
