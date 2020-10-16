/**
 * @file matmulv2_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief MatMulV2 fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_MATMULV2_FUSION__H_
#define _OPTIMIZER_FUSION_MATMULV2_FUSION__H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class MatMulV2FusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "TransposeD_MatMulV2";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_MATMULV2_FUSION__H_
