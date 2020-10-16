/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  fusedbatchnormgrad_fusion_pass.h
 *
 * @brief fusedbatchnormgrad fusion pass(fusedbatchnormgrad --> BNTrainingReduceGrad & BNTrainingUpdateGrad)
 *
 */

#ifndef FE_FUSEDBATCHNORMGRAD_FUSION_PASS_H
#define FE_FUSEDBATCHNORMGRAD_FUSION_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class FusedBatchNormGradFusionPass : public PatternFusionBasePass {
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
#endif  // FE_FUSEDBATCHNORMGRAD_FUSION_PASS_H
