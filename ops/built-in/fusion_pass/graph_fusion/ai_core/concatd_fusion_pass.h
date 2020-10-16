/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  concatd_fusion_pass.h
 *
 * @brief ConcatD fusion pass(ConcatD --> ConcatD)
 *
 */

#ifndef FE_CONCATD_FUSION_PASS_H
#define FE_CONCATD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConcatDFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ConcatD";
};

}  // namespace fe

#endif  // FE_CONCATD_FUSION_PASS_H
