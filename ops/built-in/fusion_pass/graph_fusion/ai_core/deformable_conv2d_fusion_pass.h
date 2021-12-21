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
 * \file deformable_conv2d_fusion_pass.h
 * \brief convert deformable_conv2d to group deformable_offsets + conv2d
 */
#ifndef FE_DEFORMABLE_CONV2D_FUSION_H
#define FE_DEFORMABLE_CONV2D_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class DeformableConv2dPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  bool AddOffsetDesc(ge::NodePtr& dfm_conv_node, ge::OpDescPtr& offset_desc, bool with_bias);
  bool AddConvDesc(ge::NodePtr& dfm_conv_node, ge::OpDescPtr& conv_desc, bool with_bias);
  const string fused_op_type_ = "DeformableConv2d";
};

}  // namespace fe
#endif
