/**
 * @file one_hot_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(one_hot --> one_hot_d)
 *
 */

#ifndef FE_ONEHOT_FUSION_H
#define FE_ONEHOT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class OneHotFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "OneHotD";
};

}  // namespace fe

#endif  // FE_ONEHOT_FUSION_H
