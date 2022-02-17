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
 * \file conv_add_fusion_pass.h
 * \brief conv-add fusion pass(conv-add --> conv)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_ADD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_ADD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

using namespace ge;

namespace fe {
class ConvAddFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  const string fused_op_type_ = "Conv";
  Status ConnectConvToOutput(NodePtr conv_node, NodePtr add_node);
  Status ShapeInfoCheck(OpDescPtr add_node_desc, OpDescPtr conv_node_desc,
                        size_t non_const_add_input_index, int64_t& channel_for_scalar);
  Status CheckConv3DShapeInfo(const std::vector<int64_t> &non_const_dims, const std::vector<int64_t> &const_dims);
  Status BroadcastScalar(int64_t channel_for_scalar, int64_t& new_shape, GeTensorPtr conv_bias_ptr);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_ADD_FUSION_PASS_H_
// ConvAddFusionPass
