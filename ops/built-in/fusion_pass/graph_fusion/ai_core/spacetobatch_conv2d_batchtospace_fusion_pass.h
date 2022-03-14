/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file spacetobatch_conv2d_batchtospace_fusion_pass.h
 * \brief spacetobatch_conv2d_batchtospace fusion pass(spacetobatch + conv2d + batchtospace --> conv2d)
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPACETOBATCH_CONV2D_BATCHTOSPACE_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPACETOBATCH_CONV2D_BATCHTOSPACE_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SpacetobatchConv2dBatchtospacePass : public PatternFusionBasePass {
protected:
    std::vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes) override;

private:
    Status CheckNodes(ge::NodePtr spacetobatchNode, ge::NodePtr batchtospaceNode,
        ge::NodePtr conv2dNode) const;
    Status CheckCrops(ge::NodePtr batchtospaceNode) const;
    Status CheckKernelSize(ge::OpDescPtr convDesc, int64_t dilationH, int64_t dilationW) const;
    Status CheckConvStrides(ge::OpDescPtr convDesc) const;
    Status CheckConvPads(ge::ConstGeTensorPtr spacePadPtr, ge::OpDescPtr convDesc,
        std::vector<int64_t>& convPads) const;
    Status CheckConvDilations(ge::ConstGeTensorPtr blockPtr, ge::OpDescPtr convDesc,
        std::vector<int64_t>& dilations) const;
    Status UpdateConv2dAttr(ge::NodePtr spaceNode, ge::NodePtr convNode) const;
    Status UpdateConv2dDesc(ge::OpDescPtr spaceDesc, ge::OpDescPtr batchDesc, ge::OpDescPtr convDesc) const;
    Status LinkConv2d(ge::NodePtr spacetobatchNode, ge::NodePtr batchtospaceNode, ge::NodePtr conv2dNode) const;
    Status RemoveNodes(ge::ComputeGraph& graph, ge::NodePtr spacetobatchNode, ge::NodePtr batchtospaceNode) const;

    const std::string fusedOpType_ = "spacetobatch_conv2d_batchtospace";
};
} // namespace fe

#endif
