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
 * \file conv2dbackprop_dilation_fusion_pass.h
 * \brief conv2dbackprop_input dilation fusion pass
 * dilation->conv2dbackinput or conv2dbackinput->dilation
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_DILATION_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_DILATION_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe{
class Conv2DbpInputDilationFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

private:
  Status Relink(ge::NodePtr outBackprop_node, ge::NodePtr dilation_node,
                ge::NodePtr conv2dbp_input_node, ge::NodePtr y_node,
                const int pre_achor, const int sub_achor, bool pre_dilation);
  const string FUSED_OP_TYPE = "Conv2DBackpropInputD";
};
} // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_DILATION_FUSION_PASS_H_
