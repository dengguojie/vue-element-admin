/**
 * @file tbe_fullyconnection_elemwise_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe FullyConnection-elemwise fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_FULLYCONNECTION_ELEMWISE_FUSION_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_FULLYCONNECTION_ELEMWISE_FUSION_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "common/lxfusion_json_util.h"

namespace fe {

class TbeFullyconnectionElemwiseFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeFullyconnectionElemwiseFusionPass() {}

  ~TbeFullyconnectionElemwiseFusionPass() {}

 protected:
  /*
   * @brief:  define fully connection elemwise fusion pattern
   *
   * pattern configuration limit:
   *
   * 1. FullyConnection/MatMUL +（ReLU/LeakyReLU) + (eltwise) + (AscendQuant).
   * 2. FullyConnection + (AscendDequant) + (ele-wise) + (AscendQuant).
   *
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
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  Status CheckDynamicMode(vector<ge::NodePtr>& matmulNodes, std::vector<ge::NodePtr>& fusionNodes) const;
  bool CheckMatmulDequantGeluQuantFusion(const vector<ge::NodePtr> &reluNodes);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_FULLYCONNECTION_ELEMWISE_FUSION_H
