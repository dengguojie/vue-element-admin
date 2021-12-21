/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file tbe_aipp_conv_relu_maxpooling_fusion_pass.h
 * \brief tbe aipp and convolution and relu and maxpooling ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class TbeAippConvReluMaxpoolingFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeAippConvReluMaxpoolingFusionPass() {
  }

  ~TbeAippConvReluMaxpoolingFusionPass() {
  }

 protected:
  /*
   * @brief:  define conv and relu and max_pooling input op fusion pattern
   *
   *    Aipp(optional)-->Convolution-->ElemWise(optional)-->MaxPool/Pooling
   *
   * @return TbeFusionPattern: return all valid patterns.
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
  bool CheckConvNodeValidation(const ge::NodePtr& conv_node);
  bool CheckMaxpoolNodeValidation(const ge::NodePtr& max_pool_node);
  void PoolingValidationAndFormatSet(const ge::NodePtr& aipp_node, const ge::NodePtr& conv_node,
                                     const ge::NodePtr& max_pool_node);
  const string fused_op_type_ = "FusedOp";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_CONV_RELU_MAXPOOLING_FUSION_PASS_H_
