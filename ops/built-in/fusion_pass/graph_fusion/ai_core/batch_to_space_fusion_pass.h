/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   batch_to_space_fusion_pass.h
 *
 * @brief  BatchToSpace fusion pass(BatchToSpace --> BatchToSpaceD)
 *
 */

#ifndef FE_BATCH_TO_SPACE_FUSION_H
#define FE_BATCH_TO_SPACE_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchToSpaceFusionPass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "BatchToSpaceD";
};
}  // namespace fe

#endif  // FE_BATCH_TO_SPACE_FUSION_H