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
 * \file conv3d_bp_input_elemwise_pass.h
 * \brief tbe conv3d_backprop_input + elemwise ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV3D_BACKPROP_INPUT_CONV3D_BP_INPUT_ELEMWISE_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV3D_BACKPROP_INPUT_CONV3D_BP_INPUT_ELEMWISE_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "common/lxfusion_json_util.h"

namespace fe {

class TbeConv3dDxElemwisePass : public BufferFusionPassBase {
 public:
  TbeConv3dDxElemwisePass() {
  }

  ~TbeConv3dDxElemwisePass() {
  }

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override;
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  const std::string FUSED_OP_TYPE = "FusedOp";
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV3D_BACKPROP_INPUT_CONV3D_BP_INPUT_ELEMWISE_PASS_H_