/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  concat_ext2_fusion_pass.h
 *
 * @brief ConcatExt2 fusion pass(ConcatExt2 --> ConcatExt2)
 *
 */

#ifndef FE_CONCATEXT2_FUSION_PASS_H
#define FE_CONCATEXT2_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConcatExt2FusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ConcatV2D";
};

}  // namespace fe

#endif  // FE_CONCATEXT2_FUSION_PASS_H
