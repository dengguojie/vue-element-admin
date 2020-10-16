/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 * reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 *
 */

#ifndef FE_DECONV_WEIGHT_TRANS_FUSION_H
#define FE_DECONV_WEIGHT_TRANS_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DeconvWeightTransFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    void GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
            const ge::Format &filterFormat, const vector<int64_t> &shapeNCHW,
            vector<int64_t> &dimComp, vector<int64_t> &reshapeIn,
            vector<int64_t> &transPerm, vector<int64_t> &reverseAxis,
            vector<int64_t> &reshapeOut);
    Status Relink(ge::NodePtr filterNode, ge::NodePtr dimCompNode,
            ge::NodePtr transposeNode, ge::NodePtr reformatNode,
            ge::NodePtr reshapeInNode, ge::NodePtr reverseNode,
            ge::NodePtr reshapeOutNode, ge::NodePtr deconvNode);
    const string FUSED_OP_TYPE = "Deconvolution";
};
} // namespace fe

#endif // FE_DECONV_WEIGHT_TRANS_FUSION_H
