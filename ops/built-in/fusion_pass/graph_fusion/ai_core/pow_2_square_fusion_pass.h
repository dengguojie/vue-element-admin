/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief pow2square fusion pass
 *
 */

#ifndef FE_POW_2_SQUARE_FUSION_H
#define FE_POW_2_SQUARE_FUSION_H

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"


namespace fe {
class Pow2SquareFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "Square";
};

}  // namespace fe
#endif  // FE_POW_2_SQUARE_FUSION_H

