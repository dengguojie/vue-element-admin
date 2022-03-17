/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file tbe_dx_transdata_fusion_pass.h
 * \brief (transdata) + conv2d_backprop_input + transdata fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BP_INPUT_TRANSDATA_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BP_INPUT_TRANSDATA_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeDxTransDataFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeDxTransDataFusionPass() {}

  ~TbeDxTransDataFusionPass() override {}

 protected:
  /*
   * @brief define (transdata) + dx + transdata ub fusion pattern
   *
   *    TransData + Conv2DBackPropInput + TransData
   *
   * fusion node: TransData, Conv2DBackPropInput
   *
   * @return BufferFusionPattern: return all valid patterns.
   */
  vector<BufferFusionPattern*> DefinePatterns() override;
  /*
   * @brief parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

  bool CheckTransDataFormat(const ge::NodePtr &node, const bool &is_input) const;
  bool CheckOpCube(const ge::NodePtr &cube_node) const;
  bool CheckInputNoRange(const ge::NodePtr &cube_node) const;
  void DeleteFusionNodes(const ge::NodePtr &transdata_node, vector<ge::NodePtr> &fusion_nodes,
                         const bool &erase_transdata_flag, const bool &is_input);
  const string kFusedOpType = "FusedOp";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BP_INPUT_TRANSDATA_FUSION_PASS_H
