/**
 * @file matmul_gelugrad_ub_fusion.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + relu + eltwise ops fusion pattern
 *
 * @version 1.0
 *
 */


#ifndef MATMUL_GELUGRAD_UB_FUSION_H
#define MATMUL_GELUGRAD_UB_FUSION_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class MatmulGelugradUbFusion : public BufferFusionPassBase {
public:
  explicit MatmulGelugradUbFusion() {}

  ~MatmulGelugradUbFusion() {}

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

#endif  // MATMUL_GELUGRAD_UB_FUSION_H
