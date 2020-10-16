/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   batch_to_space_nd_fusion_pass.h
 *
 * @brief  BatchToSpaceND fusion pass(BatchToSpaceND --> BatchToSpaceNDD)
 *
 */


#ifndef FE_BATCH_TO_SPACE_ND_FUSION_H
#define FE_BATCH_TO_SPACE_ND_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrBatchToSpaceNdPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "BatchToSpaceNDD";
};
}  // namespace fe
#endif  // FE_BATCH_TO_SPACE_ND_FUSION_H
