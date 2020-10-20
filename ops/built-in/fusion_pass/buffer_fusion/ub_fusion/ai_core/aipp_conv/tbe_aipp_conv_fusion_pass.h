/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file tbe_aipp_conv_fusion_pass.h
 * \brief tbe aipp convolution ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeAippConvFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeAippConvFusionPass() {
  }

  ~TbeAippConvFusionPass() {
  }

 protected:
  /*
   * @brief:  define convolution and single input op fusion pattern
   *
   * pattern configuration limit:
   * 1. total min value must be 1 for all head candidated desc.
   * 2. any head candidated desc max value must be 1.
   * 3. output desc can not be itself.
   *
   *    1) Aipp-->Convolution
   *
   * fusion node: Aipp, Convolution
   *
   * @return BufferFusionPattern: return all valid patterns.
   */
  vector<BufferFusionPattern*> DefinePatterns() override;

  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  const string fused_op_type_ = "FusedOp";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_FUSION_PASS_H_
