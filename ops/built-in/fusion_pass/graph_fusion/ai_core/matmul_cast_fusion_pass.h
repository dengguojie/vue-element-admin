/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief matmul cast fusion (Matmul--Cast)
 *
 */
#ifndef FE_FUSION_MATMUL_CAST_FUSION_PASS_H_
#define FE_FUSION_MATMUL_CAST_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class MatmulCastFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    Status LinkOutputEdgeWithoutControl(ge::NodePtr oldNode,
                                        ge::NodePtr newNode);
    Status IsMatch(ge::NodePtr matmulNode, ge::NodePtr castNode);
    Status DoFusion(ge::NodePtr matmulNode);
    const std::string CAST = "Cast";
    const string FUSED_OP_TYPE = "MatMul/MatMulV2";
};

}  // namespace fe
#endif  // FE_FUSION_MATMUL_CAST_FUSION_PASS_H_
