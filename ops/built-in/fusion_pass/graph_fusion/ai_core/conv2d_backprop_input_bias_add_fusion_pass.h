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
 * \file conv2d_backprop_input_bias_add_fusion_pass.h
 * \brief pseudo fusion pass of "dx + bias_add"
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_INPUT_BIAS_ADD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_INPUT_BIAS_ADD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2DbpInputBiasAddFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

private:
  Status convert_dx_to_transpose(ge::ComputeGraph &graph, Mapping &mapping,
                                 vector<ge::NodePtr> &fusion_nodes);
  Status connect_edges(ge::NodePtr &conv_node, ge::NodePtr &bias_node,
                       ge::NodePtr &bias_const_node, ge::NodePtr &conv2d_transpose_d);
  void set_in_out_op_and_attr(ge::OpDescPtr &conv_op, ge::OpDescPtr &bias_const_op,
                              ge::OpDescPtr &conv2d_transpose_d_op);
  const string FUSED_OP_TYPE = "Conv2DBackpropInputD";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_INPUT_BIAS_ADD_FUSION_PASS_H_
