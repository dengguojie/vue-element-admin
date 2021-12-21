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
 * \file conv2d_bp_input_elemwise_pass.h
 * \brief tbe conv2d_backprop_input + elemwise ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BACKPROP_INPUT_CONV2D_BP_INPUT_ELEMWISE_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BACKPROP_INPUT_CONV2D_BP_INPUT_ELEMWISE_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "common/lxfusion_json_util.h"

namespace fe {

class TbeDxElemwisePass : public BufferFusionPassBase {
 public:
  TbeDxElemwisePass() {
  }

  ~TbeDxElemwisePass() override {
  }

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override;
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  const std::string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV2D_BACKPROP_INPUT_CONV2D_BP_INPUT_ELEMWISE_PASS_H_
