/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Split fusion pass(Split --> SplitD)
 *
 */

#ifndef FE_SPLIT_FUSION_PASS_H
#define FE_SPLIT_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SplitFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE = "SplitVD";
};

}  // namespace fe

#endif  // FE_SPLIT_FUSION_PASS_H
