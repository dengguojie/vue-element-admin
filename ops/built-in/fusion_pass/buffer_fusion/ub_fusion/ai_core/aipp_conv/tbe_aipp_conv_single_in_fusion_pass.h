/**
 * @file tbe_aipp_conv_single_in_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe aipp convolution and element-wise ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef TBE_AIPP_CONV_SINGLE_IN_FUSION_H
#define TBE_AIPP_CONV_SINGLE_IN_FUSION_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeAippConvSingleInFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeAippConvSingleInFusionPass()  {}

  ~TbeAippConvSingleInFusionPass() {}

 protected:
  /*
   * @brief:  define convolution and single input op fusion pattern
   *
   * pattern configuration limit:
   * 1. total min value must be 1 for all head candidated desc.
   * 2. any head candidated desc max value must be 1.
   * 3. output desc can not be itself.
   *
   *    1) Convolution-->ElemWise_1-->ElemWise_2
   *
   * fusion node: ElemWise_1, ElemWise_2, Convolution
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
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // TBE_AIPP_CONV_SINGLE_IN_FUSION_H
