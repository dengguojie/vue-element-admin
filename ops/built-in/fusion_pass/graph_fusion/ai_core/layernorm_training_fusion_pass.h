/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file layernorm_training_fusion_pass.h
 * \brief fused Add(add), Mul(three), Sub of structure:
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LayerNormTrainingFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status RemoveSmalleNodes(ge::ComputeGraph& graph, const ge::NodePtr& mean_node1,
                           const ge::NodePtr& square_difference_node, const ge::NodePtr& mean_node2,
                           const ge::NodePtr& add_node1, const ge::NodePtr& rsqrt_node,
                           const ge::NodePtr& mul_node1, const ge::NodePtr& mul_node2,
                           const ge::NodePtr& mul_node3, const ge::NodePtr& sub_node,
                           const ge::NodePtr& add_node2);

  Status AddTensorDescForLn(const ge::OpDescPtr& ln_opdesc, const ge::GeTensorDesc& x_tensor,
                            const ge::GeTensorDesc& gamma_tensor,
                            const ge::GeTensorDesc& beta_tensor, const ge::GeTensorDesc& ln_out_tensor,
                            const ge::GeTensorDesc& mean_out_tensor,
                            const ge::GeTensorDesc& varience_out_tensor);

  const string FUSED_OP_TYPE = "LayerNorm";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_
