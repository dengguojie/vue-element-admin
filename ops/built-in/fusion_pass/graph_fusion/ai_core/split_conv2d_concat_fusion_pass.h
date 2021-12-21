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
 * \file split_conv2d_concat_fusion_pass.h
 * \brief convert split+conv2d+concat to group conv2d
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPLIT_CONV2D_CONCAT_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPLIT_CONV2D_CONCAT_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class SplitConv2dConcatPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  bool AddConcatDesc(ge::NodePtr& split_node, ge::NodePtr& ccat_node, std::vector<ge::OpDescPtr>& const_desc);
  bool LinkGroupConv2d(ge::NodePtr& groupConv, ge::NodePtr& split_node, ge::NodePtr& ccat_node,
                       std::vector<ge::NodePtr>& const_ccat);
  bool AnalyzeMidLayer(ge::Node::Vistor<ge::NodePtr>& spt_output, ge::OpDescPtr& conv_gp_desc);
  bool VerifySptCcatAxis(ge::OpDescPtr& conv_desc, ge::NodePtr& split_node, ge::NodePtr& ccat_node);
  bool LinkNewConcat(ge::ComputeGraph& graph, ge::NodePtr& split_node, std::vector<ge::NodePtr>& const_ccat,
                     std::vector<ge::NodePtr>& const_dim);
  bool UpdateConv2dDesc(ge::OpDescPtr& conv_desc, ge::NodePtr& split_node, ge::NodePtr& ccat_node);
  const string fused_op_type_ = "Split_Conv2D_ConcatV2";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SPLIT_CONV2D_CONCAT_FUSION_PASS_H_
