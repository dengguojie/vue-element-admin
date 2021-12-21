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
 * \file conv_to_fullyconnection_fusion_pass.h
 * \brief fuse conv to fullyconnection
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_TO_FULLYCONNECTION_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_TO_FULLYCONNECTION_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConvToFullyConnectionFusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector <ge::NodePtr> &fusionNodes) override;
private:
    Status CheckFusionParm(ge::NodePtr convNode);
    void RefreshBiasNodeFromSubgraphToMajorgraph(ge::NodePtr convNode);
    Status CreateReshapeNode(ge::ComputeGraph &graph, const ge::OutDataAnchorPtr &out_anchor,
                             const vector <int64_t> &shape, ge::NodePtr &shape_node);
    Status InsertNode(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst,
                      ge::NodePtr &new_node);
    Status CheckHWCEqual(const ge::GeTensorDesc &xTensor, const ge::GeTensorDesc &filterTensor);
    int64_t GetDimByAxisName(const ge::GeTensorDesc &tensor, const string &axis);
    int32_t GetIndexByAxisName(const ge::GeTensorDesc &tensor, const string &axis);
    const string FUSED_OP_TYPE = "Conv2D";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_TO_FULLYCONNECTION_FUSION_PASS_H_
