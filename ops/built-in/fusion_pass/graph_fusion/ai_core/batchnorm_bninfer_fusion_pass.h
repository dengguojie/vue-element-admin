/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief BatchNorm BnInfer fusion pass
 *
 */

#ifndef FE_OPTIMIZER_FUSION_BATCHNORM_BNINFER_FUSION_H_
#define FE_OPTIMIZER_FUSION_BATCHNORM_BNINFER_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchNormBnInferFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "BNInference";
};
}  // namespace fe
#endif  // FE_OPTIMIZER_FUSION_BATCHNORM_BNINFER_FUSION_H_
