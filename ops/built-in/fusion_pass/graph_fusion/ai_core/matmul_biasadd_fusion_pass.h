/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief matmul biasadd fusion pass(matmul --> biasadd)
 *
 */
#ifndef FE_MATMUL_BIASADD_FUSION_PASS_H
#define FE_MATMUL_BIASADD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class MatMulBiasAddFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const std::string CONSTANTOP = "Constant";
    const std::string CONSTANT = "Const";
    const string FUSED_OP_TYPE = "MatMul/MatMulV2";
};
}  // namespace fe

#endif  // FE_MATMUL_BIASADD_FUSION_PASS_H
