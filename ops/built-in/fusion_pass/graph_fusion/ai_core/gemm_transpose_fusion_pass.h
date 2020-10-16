/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief gemm transpose fusion pass
 *
 */

#ifndef FE_GEMM_TRANS_FUSION_H
#define FE_GEMM_TRANS_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class GemmTransFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    Status Relink(ge::NodePtr aNode, ge::NodePtr transposeANode,
                         ge::NodePtr gemmNode, const int Anchor);
    const string FUSED_OP_TYPE = "GEMM";
};
} // namespace fe

#endif // FE_GEMM_TRANS_FUSION_H
