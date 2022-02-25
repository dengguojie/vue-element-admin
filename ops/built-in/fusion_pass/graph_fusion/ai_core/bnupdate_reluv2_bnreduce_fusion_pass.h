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
 * \file bnupdate_reluv2_bnreduce_fusion_pass.h
 * \brief convert bnupdate+reluv2+conv2d+bnreduce to fusedbn2reluconvbn1
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BNUPDATE_RELUV2_Conv2D_BNREDUCE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BNUPDATE_RELUV2_Conv2D_BNREDUCE_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BNupdateReluV2Conv2DBNreducePass : public PatternFusionBasePass {
protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

private:
    bool CheckBnupdateNodeOutSingleReferred(const std::vector<ge::NodePtr>& node_list,
                                            const int idx,
                                            const std::vector<size_t>& ir_inputs,
                                            const std::vector<size_t>& expect_ref);
    bool AnalyzeLayers(const std::vector<ge::NodePtr> &node_list);
    bool ReplaceFusedDescName(const std::vector<std::string>& ori_names,
                              std::map<std::string, std::vector<std::string>>& ori_to_new,
                              const string& name_attr,
                              std::map<std::string, uint32_t>& new_name);
    bool UpdateDesc(const std::vector<ge::NodePtr> &node_list, ge::OpDescPtr fused_desc,
                    std::vector<std::string> fused_inputs, std::vector<std::string> fused_outputs);
    bool AddFusedDesc(const std::vector<ge::NodePtr> &node_list, ge::OpDescPtr fused_desc);
    bool LinkValidInputDataAnchor(ge::Node::Vistor<ge::InDataAnchorPtr>& input_anchors,
                                  const std::map<std::string, size_t>& node_name,
                                  ge::NodePtr& fused_node,
                                  uint32_t& inputIdx);
    bool LinkValidOutputDataAnchor(ge::Node::Vistor<ge::OutDataAnchorPtr>& output_anchors,
                                   const std::map<std::string, size_t>& node_name,
                                   ge::NodePtr& fused_node,
                                   uint32_t& outputIdx);
    bool LinkNewNode(const std::vector<ge::NodePtr> &node_list, ge::NodePtr fused_node);
    const std::string fused_op_type_ = "BNupdate_ReluV2_Conv2D_BNreduce";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BNUPDATE_RELUV2_Conv2D_BNREDUCE_FUSION_PASS_H_