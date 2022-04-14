/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file randomdsa_fusion_pass.h 
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_DSA_CORE_RANDOMDSA_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_DSA_CORE_RANDOMDSA_FUSION_PASS_H_
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe{
    class RandomDsaFusionPass : public PatternFusionBasePass {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

    private:
        Status FlatternShape(ge::ComputeGraph &graph, ge::NodePtr &fusion_node, int32_t index);
        Status CreateConstOperator(ge::ComputeGraph &graph,ge::NodePtr &fusion_node, ge::NodePtr &const_node);
        Status CreateConstOperatorNormal(ge::ComputeGraph &graph, const float mean, const float stddev,
                                         ge::NodePtr &const_node_mean, ge::NodePtr &const_node_stddev);
        const string FUSED_OP_TYPE_PROD = "ReduceProd";
        const string FUSED_OP_TYPE = "RandomDsa";
    };
}
# endif  //  OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_DSA_CORE_RANDOMDSA_FUSION_PASS_H_
