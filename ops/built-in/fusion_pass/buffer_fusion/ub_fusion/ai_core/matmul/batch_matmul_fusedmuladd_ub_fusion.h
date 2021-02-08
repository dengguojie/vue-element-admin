/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file batch_matmul_fusedmuladd_ub_fusion.h
 * \brief batch_matmul and FusedMulAdd ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_BATCH_MATMUL_FUSEDMULADD_UB_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_BATCH_MATMUL_FUSEDMULADD_UB_FUSION_H_

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include <vector>

namespace fe {

class TbeBatchMatmulFusedMulAddFusionPass : public BufferFusionPassBase {
  public:
    explicit TbeBatchMatmulFusedMulAddFusionPass() {}

    ~TbeBatchMatmulFusedMulAddFusionPass() {}

  protected:
    vector<BufferFusionPattern*> DefinePatterns() override;
    Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

  private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TBE_MATMUL_ELEMWISE_H_
