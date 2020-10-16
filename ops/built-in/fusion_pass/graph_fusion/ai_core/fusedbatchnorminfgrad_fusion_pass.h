/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fusedbatchnormgrad fusion pass(min --> max)
 *
 */

#ifndef FE_FUSEDBATCHNORMINFGRAD_FUSION_PASS_H
#define FE_FUSEDBATCHNORMINFGRAD_FUSION_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class FusedBatchNormInfGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping &mapping);
  const string FUSED_OP_TYPE = "BNTrainingReduceGrad_BNTrainingUpdateGrad";
};
}  // namespace fe
#endif  // FE_FUSEDBATCHNORMINFGRAD_FUSION_PASS_H
