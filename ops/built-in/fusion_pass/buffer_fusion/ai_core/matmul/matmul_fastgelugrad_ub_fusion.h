/**
 * @file matmul_fastgelugrad_ub_fusion.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + relu + eltwise ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_MATMUL_MATMUL_FASTGELUGRAD_UB_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_MATMUL_MATMUL_FASTGELUGRAD_UB_FUSION_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class MatmulFastGelugradUbFusion : public BufferFusionPassBase {
 public:
  explicit MatmulFastGelugradUbFusion() {
  }

  ~MatmulFastGelugradUbFusion() {
  }

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override;
  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusionNodes) override;
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_MATMUL_MATMUL_FASTGELUGRAD_UB_FUSION_H_
