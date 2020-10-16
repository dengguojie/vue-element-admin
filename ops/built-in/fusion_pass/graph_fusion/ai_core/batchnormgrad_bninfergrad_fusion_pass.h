/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  batchnormgrad_bninfergrad_fusion_pass.h
 *
 * @brief BatchNormGrad BnInferGrad fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_BATCHNORMGRAD_BNINFERGRAD_FUSION_H_
#define _OPTIMIZER_FUSION_BATCHNORMGRAD_BNINFERGRAD_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchNormGradBnInferGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "BNInferGrad";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_BATCHNORMGRAD_BNINFERGRAD_FUSION_H_
