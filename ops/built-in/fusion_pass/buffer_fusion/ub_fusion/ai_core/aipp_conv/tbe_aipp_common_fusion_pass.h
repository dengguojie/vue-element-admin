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
 * \file tbe_aipp_common_fusion_pass.h
 * \brief tbe aipp convolution common ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_COMMON_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_COMMON_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeAippCommonFusionPass : public BufferFusionPassBase {
 public:
  TbeAippCommonFusionPass() {}

  ~TbeAippCommonFusionPass() {}

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

 private:
  const string fused_op_type_ = "AippCommonFusedOp";

  void AddRemovingReluNodes(ge::NodePtr remove_node, const std::vector<ge::NodePtr> &elemwise_nodes,
                            std::vector<ge::NodePtr> &remove_nodes);
  void FilterElemwiseNodes(std::vector<ge::NodePtr> &elemwise_nodes, std::vector<ge::NodePtr> &quant_nodes,
                           std::vector<ge::NodePtr> &strided_write_nodes, std::vector<ge::NodePtr> &fusion_nodes);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_COMMON_FUSION_PASS_H_
