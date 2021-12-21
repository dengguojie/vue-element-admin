/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file matmul_confusiontranspose_ub_fusion.h
 * \brief  tbe matmul + confusiontransposed ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class MatmulConfusiontransposeUbFusion : public BufferFusionPassBase {
 public:
  explicit MatmulConfusiontransposeUbFusion() {
  }

  ~MatmulConfusiontransposeUbFusion() {
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

 private:
  const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_CONFUSIONTRANSPOSE_UB_FUSION_H_
