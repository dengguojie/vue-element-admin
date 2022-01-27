/*
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
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2DBACKPROPINPUT_TBE_DW_TRANSDATA_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2DBACKPROPINPUT_TBE_DW_TRANSDATA_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"


namespace fe {

class TbeDwTransDataFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeDwTransDataFusionPass() {}

  ~TbeDwTransDataFusionPass() override {}

 protected:
  /*
   * @brief: define transdata_dw fusion pattern
   *
   *  TransData    TransData
   *      \           /
   *       \         /
   *    Conv2DBackporpFilter
   * pattern limit:
   *       1. both  inputs of dw must be with trans_data
   *       2. format is transfered from NCHW to NC1HWC0 by trans_data
   * @return BufferFusionPattern: return all valid patterns
   */
  vector<BufferFusionPattern *> DefinePatterns() override;

  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector <ge::NodePtr> &fusion_nodes) override;

 private:
  bool CheckDwSupport(const vector<ge::NodePtr> &dw_nodes) const;
  bool CheckTransdataSupport(const vector<ge::NodePtr> &dw_nodes, const vector<ge::NodePtr> &transdata_nodes,
                             vector<ge::NodePtr> &fusion_nodes);
  bool CheckSupportTrans(const ge::NodePtr &node) const;
  bool CheckUnlimitedRange(const ge::NodePtr &node) const;
};
}


#endif //OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2DBACKPROPINPUT_TBE_DW_TRANSDATA_FUSION_PASS_H
