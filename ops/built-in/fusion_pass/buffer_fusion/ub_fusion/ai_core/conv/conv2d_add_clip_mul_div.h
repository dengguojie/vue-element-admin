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
 * \file conv2d_add_clip_mul_div.h
 * \brief tbe conv2d + add + clip + mul + div  ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_ADD_CLIP_MUL_DIV_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_ADD_CLIP_MUL_DIV_PASS_H_

#include <vector>
#include "common/lxfusion_json_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class Conv2dAddClipMulDivFusionPass : public BufferFusionPassBase {
public:
  Conv2dAddClipMulDivFusionPass() {}
  ~Conv2dAddClipMulDivFusionPass() {}
  Status CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info) override;

protected:

  vector<BufferFusionPattern*> DefinePatterns() override;

private:
  const string fused_op_type_ = "Conv2dAddClipMulDiv";
};

}  // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_CONV_CONV2D_ADD_CLIP_MUL_DIV_PASS_H_