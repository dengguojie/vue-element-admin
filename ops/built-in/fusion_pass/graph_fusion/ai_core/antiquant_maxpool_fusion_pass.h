/**
 * @file antiquant_maxpool_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief antiquant maxpool fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_ANTIQUANT_MAXPOOL_FUSION_H_
#define _OPTIMIZER_FUSION_ANTIQUANT_MAXPOOL_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AntiQuantMaxPoolFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "AscendAntiQuant";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_ANTIQUANT_MAXPOOL_FUSION_H_
