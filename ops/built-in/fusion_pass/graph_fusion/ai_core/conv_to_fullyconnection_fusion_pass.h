/**
 * @file conv_to_fullyconnection_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief fuse conv to fullyconnection
 *
 * @version 1.0
 *
 */
#ifndef _FE_CONV_TO_FULLYCONNECTION_FUSION_H_
#define _FE_CONV_TO_FULLYCONNECTION_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConvToFullyConnectionFusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;

    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) override;

private:
    Status CheckFusionParm(ge::NodePtr convNode);
    Status CheckHWCEqual(const ge::GeTensorDesc &xTensor,
                         const ge::GeTensorDesc &filterTensor);
    int64_t GetDimByAxisName(const ge::GeTensorDesc &tensor, const string &axis);
    const string FUSED_OP_TYPE = "Conv2D";
};
}  // namespace fe
#endif  // _FE_CONV_TO_FULLYCONNECTION_FUSION_H_
