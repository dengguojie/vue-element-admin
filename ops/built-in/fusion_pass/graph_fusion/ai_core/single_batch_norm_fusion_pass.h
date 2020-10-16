/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Single BatchMorm fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_SINGLE_BATCHNORM_FUSION_H_
#define _OPTIMIZER_FUSION_SINGLE_BATCHNORM_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SingleBatchNormFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE = "BNTrainingReduce_BNTrainingUpdateV3";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_SINGLE_BATCHNORM_FUSION_H_
