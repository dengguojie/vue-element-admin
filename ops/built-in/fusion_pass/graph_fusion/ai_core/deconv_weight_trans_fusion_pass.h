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
 * \file deconv_weight_trans_fusion_pass.h
 * \brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 *   reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DeconvWeightTransFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  int64_t GetGroups(ge::OpDescPtr &deconv_desc);
  void GetShapeUsedByIntermediateProcessInDeconvWeightTrans(const ge::Format& filter_format,
                                                            const vector<int64_t>& shape_NCHW, vector<int64_t>& complement_dimension,
                                                            vector<int64_t>& reshape_in, vector<int64_t>& permute_shape,
                                                            vector<int64_t>& reverse_axis, vector<int64_t>& reshape_out);
  static Status Relink(ge::NodePtr filter_node, ge::NodePtr complement_dimension_node, ge::NodePtr transpose_node, ge::NodePtr reformat_node,
                ge::NodePtr reshape_in_node, ge::NodePtr reverse_node, ge::NodePtr reshape_out_node, ge::NodePtr deconv_node);
  static Status GenerateTransposeNode(ge::ComputeGraph& graph, ge::GeTensorDesc& previous_out_desc,
                               ge::GeTensorDesc& next_in_desc, const vector<int64_t>& perm, ge::NodePtr& transpose_node,
                               const std::string& basename);
  static Status GenerateReshapeNode(ge::ComputeGraph& graph, ge::GeTensorDesc& previous_out_desc,
                                    ge::GeTensorDesc& next_in_desc, const vector<int64_t>& shape,
                                    ge::NodePtr& shape_node, const std::string& name, const std::string& basename);
  static Status GenerateReverseNode(ge::ComputeGraph& graph, ge::GeTensorDesc& previous_out_desc,
                                    ge::GeTensorDesc& next_in_desc, const vector<int64_t>& axis,
                                    ge::NodePtr& reverse_node, const std::string& basename);
  static Status GenerateReFormatNode(ge::ComputeGraph& graph, ge::GeTensorDesc& previous_out_desc,
                                     ge::GeTensorDesc& next_in_desc, const ge::Format& format,
                                     ge::NodePtr& reformat_node, const std::string& basename);
  static const string FUSED_OP_TYPE;
  };
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_