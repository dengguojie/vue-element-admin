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

/* !
 * \file grid_sampler_fusion_pass.h
 * \brief grid_sampler fusion pass
 */

#ifndef FE_GRID_SAMPLER_FUSION_H
#define FE_GRID_SAMPLER_FUSION_H

#include <string>
#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class GridSamplerFusionPass : public PatternFusionBasePass {
    protected:
    std::vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, std::vector<ge::NodePtr> &new_nodes) override;

    private:
    std::vector<int64_t> input_dims;
    std::vector<int64_t> output_dims;
    ge::DataType inputX_type = ge::DT_FLOAT;
    ge::DataType grid_type = ge::DT_FLOAT;

    void GetNodeInfo(ge::NodePtr node);
    void SetTensorDesc(ge::GeTensorDesc &tensorDesc, const std::vector<int64_t> &dims, const ge::Format &format,
                       const ge::DataType &dtype) const;
    void AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string &name, const vector<int64_t> &dims,
                          const ge::Format &format, const ge::DataType &dtype) const;
    void AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string &name, const vector<int64_t> &dims,
                           const ge::Format &format, const ge::DataType &dtype) const;
    ge::NodePtr AddNewNode(ge::ComputeGraph &graph, ge::OpDescPtr &op_desc, std::vector<ge::NodePtr> &new_nodes) const;

    Status AddGridUnnormalNode(ge::NodePtr grid_sampler, ge::NodePtr &unnormal_part, ge::NodePtr &concat,
                               ge::ComputeGraph &graph, std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddImageUnfoldNode(ge::NodePtr grid_sampler, ge::NodePtr unnormal, ge::ComputeGraph &graph,
                                   std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddRightMatmulNode(ge::NodePtr grid_sampler, ge::NodePtr broadcast_node, ge::ComputeGraph &graph,
                                   std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddBroadCastNode(ge::NodePtr grid_sampler, ge::NodePtr concat, ge::ComputeGraph &graph,
                                 std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddLeftMatmulNode(ge::NodePtr grid_sampler, ge::NodePtr unfold_node, ge::NodePtr rbmm,
                                  ge::ComputeGraph &graph, std::vector<ge::NodePtr> &new_nodes) const;
    // unnormal node
    ge::NodePtr AddGridUnnormalPartNode(ge::NodePtr grid_sampler, ge::ComputeGraph &graph,
                                        std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddSplitNode(ge::NodePtr grid_sampler, ge::NodePtr unnormal_part, ge::ComputeGraph &graph,
                             std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddVmulNode(ge::NodePtr grid_sampler, ge::NodePtr split_node, ge::ComputeGraph &graph,
                            std::vector<ge::NodePtr> &new_nodes) const;
    ge::NodePtr AddConcatNode(ge::NodePtr grid_sampler, ge::NodePtr vmul_node, ge::NodePtr unnormal_part,
                              ge::ComputeGraph &graph, std::vector<ge::NodePtr> &new_nodes) const;
};
}  // namespace fe

#endif  // FE_GRID_SAMPLER_FUSION_H
