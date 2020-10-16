/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief all const2attr pass.
 *
 * @version 1.0
 *
 */
#ifndef BUILT_IN_FUSION_PASS_APPLY_ADDOUTPUT_FUSION_H
#define BUILT_IN_FUSION_PASS_APPLY_ADDOUTPUT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AddOutputFusionPass : public PatternFusionBasePass {
 protected:
  Status Run(ge::ComputeGraph &graph,
             OpsKernelInfoStorePtr opsKernelInfoStorePtr) override;
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

};
}  // namespace fe

#endif  // FE_CONST2ATTR_FUSION_H