/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @file  concat_v2d_fusion_pass.h
 *
 * @brief Concatv2d fusion pass(multi Concatv2d --> single Concatv2d)
 *
 */

#ifndef FE_CONCATV2D_FUSION_PASS_H
#define FE_CONCATV2D_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Concatv2dFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &fusionNodes) override;

private:
    bool CheckConcatValid(ge::NodePtr node, ge::Format format,
            ge::GeShape shape, int32_t dimNum);
    Status PatternParse(ge::NodePtr concatv2dNode,
            vector<ge::NodePtr> &fusedInputNodes,
            vector<ge::NodePtr> &concatNodes);
    const string FUSED_OP_TYPE = "ConcatV2D";
};

}  // namespace fe

#endif  // FE_CONCATV2D_FUSION_PASS_H
