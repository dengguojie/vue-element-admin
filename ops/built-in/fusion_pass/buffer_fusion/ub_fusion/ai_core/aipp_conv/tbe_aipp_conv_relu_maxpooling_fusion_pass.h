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

#ifndef TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H
#define TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class TbeAippConvReluMaxpoolingFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeAippConvReluMaxpoolingFusionPass() {}

  ~TbeAippConvReluMaxpoolingFusionPass() {}

 protected:
  /*
   * @brief:  define conv and relu and max_pooling input op fusion pattern
   *
   *    Aipp-->Convolution-->ElemWise(optional)-->MaxPool/Pooling
   *
   * @return TbeFusionPattern: return all valid patterns.
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
    bool CheckConvPoolNodeValidation(ge::NodePtr convNode);
    bool CheckMaxpoolNodeValidation(ge::NodePtr maxPoolNode);
    const string FUSED_OP_TYPE = "FusedOp";
};
}  // namespace fe

#endif  // TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H
