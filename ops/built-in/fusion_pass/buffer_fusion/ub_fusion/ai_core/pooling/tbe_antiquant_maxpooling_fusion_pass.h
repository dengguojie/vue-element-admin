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

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_POOLING_TBE_ANTIQUANT_MAXPOOLING_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_POOLING_TBE_ANTIQUANT_MAXPOOLING_FUSION_PASS_H_

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class AntiquantMaxpoolingFusionPass : public BufferFusionPassBase {
public:
  AntiquantMaxpoolingFusionPass() {}

  ~AntiquantMaxpoolingFusionPass() {}

protected:
  /*
   * @brief:  define convolution common op fusion pattern
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

  /*
   * @brief: Set split info for patterns
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: void
   */
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr>& fusion_nodes);

private:
  const string fused_op_type_ = "FusedOp";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_POOLING_TBE_ANTIQUANT_MAXPOOLING_FUSION_PASS_H_
