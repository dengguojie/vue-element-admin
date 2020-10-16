/**
 * @file tbe_matmul_elemwise.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe matmul and element-wise ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef TBE_MATMUL_ELEMWISE_FUSION_H
#define TBE_MATMUL_ELEMWISE_FUSION_H

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include <vector>

namespace fe {

class TbeMatmulElemwiseFusionPass : public BufferFusionPassBase {
public:
  explicit TbeMatmulElemwiseFusionPass() {}

  ~TbeMatmulElemwiseFusionPass() {}

protected:
  /*
   * @brief:  define Matmul and element-wise op fusion pattern
   *
   *   Matmul + ElemWise
   *
   * fusion node:  Matmul, ElemWise
   *
   * @return BufferFusionPattern: return all valid patterns.
   */
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

} // namespace fe

#endif // TBE_MATMUL_ELEMWISE_FUSION_H
