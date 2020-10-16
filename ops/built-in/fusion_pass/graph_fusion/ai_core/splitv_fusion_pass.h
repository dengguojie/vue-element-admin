/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief SplitV fusion pass(SplitV --> SplitVD)
 *
 */

#ifndef FE_SPLITV_FUSION_PASS_H
#define FE_SPLITV_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SplitVFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "SplitVD";
};

}  // namespace fe

#endif  // FE_SPLITV_FUSION_PASS_H
