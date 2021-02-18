/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_TBE_CONV_REQUANT_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_TBE_CONV_REQUANT_PASS_H_

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class ConvRequantFusionPass : public BufferFusionPassBase {
public:
  ConvRequantFusionPass() {}

  ~ConvRequantFusionPass() {}

protected:
  /*
   * @brief:  define convolution common op fusion pattern
   *
   * @return BufferFusionPattern: return all valid patterns.
   */
  vector<BufferFusionPattern*> DefinePatterns() override;

private:
  const string fused_op_type_ = "FusedOp";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_TBE_CONV_REQUANT_PASS_H_
