/**
 * @file matmul_confusiontranspose_ub_fusion.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe matmul + confusiontransposed ops fusion pattern
 *
 * @version 1.0
 *
 */


#ifndef MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H
#define MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class MatmulConfusiontransposeUbFusion : public BufferFusionPassBase {
public:
  explicit MatmulConfusiontransposeUbFusion() {}

  ~MatmulConfusiontransposeUbFusion() {}

protected:
  vector<BufferFusionPattern *> DefinePatterns() override;
  /*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
  Status GetFusionNodes(const BufferFusionMapping &mapping,
                        vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H
