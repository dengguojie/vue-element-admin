/**
 * @file tbe_conv_bnreduce_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe convolution and BNReduce ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef TBE_CONV2D_QUANT_STRIDEDWRITE_FUSION_PASS_H
#define TBE_CONV2D_QUANT_STRIDEDWRITE_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class TbeConv2dQuantStridewriteFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeConv2dQuantStridewriteFusionPass() {}

  ~TbeConv2dQuantStridewriteFusionPass() {}

 protected:
  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
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
}  // namespace fe

#endif  // TBE_CONV2D_QUANT_STRIDEDWRITE_FUSION_PASS_H
