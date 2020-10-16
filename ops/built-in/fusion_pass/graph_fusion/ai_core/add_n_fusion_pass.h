/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   add_n_fusion_pass.h
 *
 * @brief  AddN fusion pass(ADDN --> ADDN)
 *
 */

#ifndef FE_ADDN_FUSION_PASS_H
#define FE_ADDN_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AddNFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "AddN";
};

}  // namespace fe

#endif  // FE_ADDN_FUSION_PASS_H
