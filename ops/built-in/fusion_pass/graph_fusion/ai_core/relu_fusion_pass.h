/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief relu fusion pass(src --> relu)
 *
 */
#ifndef FE_RELU_FUSION_PASS_H
#define FE_RELU_FUSION_PASS_H

#include <set>
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ReluFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  Status DoFusion(ge::NodePtr src_node, ge::NodePtr relu_node);
  const string FUSED_OP_TYPE = "Convolution/Eltwise/FullConnection/Add";
};

}  // namespace fe

#endif  // FE_RELU_FUSION_PASS_H
