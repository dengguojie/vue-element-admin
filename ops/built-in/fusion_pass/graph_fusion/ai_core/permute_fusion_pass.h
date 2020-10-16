/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Permute fusion pass(Permute --> TransposeD)
 *
 */

#ifndef FE_PERMUTE_FUSION_PASS_H
#define FE_PERMUTE_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class PermuteFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE = "TransposeD";
};

}  // namespace fe

#endif  // FE_PERMUTE_FUSION_PASS_H
