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

/*!
 * \file conv2d_writeselect_stridewrite_pass.h
 * \brief tbe conv2d + write_select + stride_write ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_WRITESELECT_STRIDEWRITE_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_WRITESELECT_STRIDEWRITE_PASS_H_

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "common/lxfusion_json_util.h"

namespace fe {

class TbeConv2dWrtselStridewrtPass : public BufferFusionPassBase {
 public:
  explicit TbeConv2dWrtselStridewrtPass() {
  }

  ~TbeConv2dWrtselStridewrtPass() {
  }

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override;
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  const string fused_op_type_ = "FusedOp";
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_WRITESELECT_STRIDEWRITE_PASS_H_
