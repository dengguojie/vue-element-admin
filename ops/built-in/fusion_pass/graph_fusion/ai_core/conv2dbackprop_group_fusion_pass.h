/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file conv2dbackprop_group_fusion_pass.h
 * \brief groups Conv2D & Conv2DBackpropInputD & Conv2DBackpropFilterD fusion
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_GROUP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_GROUP_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2dbackpropFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status IsMatch(ge::NodePtr& Conv2DBackpropInputNode, ge::NodePtr& Conv2DBackpropFilterNode, ge::NodePtr& Conv2DNode);
  Status CheckValidation(ge::OpDescPtr Conv2DDesc, ge::OpDescPtr Conv2DBackpropInputDesc,
                         ge::OpDescPtr Conv2DBackpropFilterDesc, int64_t& Conv2DGroups);
  Status ProcessDepthwiseConv(ge::OpDescPtr& Conv2DDesc, ge::OpDescPtr& Conv2DBackpropInputDesc,
                              ge::OpDescPtr& Conv2DBackpropFilterDesc, int64_t Conv2DGroups);
  Status ProcessGroupConvFusion(ge::ComputeGraph& graph, ge::NodePtr Conv2DNode, ge::NodePtr Conv2DBackpropInputNode,
                                ge::NodePtr Conv2DBackpropFilterNode, int64_t Conv2DGroups);
  Status GetGroups(ge::OpDescPtr srcDesc, int64_t& groups);

  Status ParseConvNodeChannelIdx(ge::GeTensorDesc& ConvTensordesc, size_t& ConvChannelIdx);

  Status ParseConvNodeNumberIdx(ge::GeTensorDesc& ConvTensordesc, size_t& ConvNumberIdx);

  Status ParseConvNodeChannel(ge::GeTensorDesc& ConvTensordesc, int64_t& ConvChannel);

  bool GenerateSplitNode(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, int64_t groups, ge::NodePtr& splitNode,
                         ge::GeTensorDesc& splitOutDesc, size_t dimIdx, uint32_t anchorIdx);

  bool GenerateNewConvNodes(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, const ge::GeTensorDesc& splitOutDesc,
                            vector<ge::NodePtr>& newConvNodes, ge::GeTensorDesc& newConvOutDesc, int64_t groups,
                            size_t dimIdx, uint32_t anchorIdx);

  bool GenerateConcatNode(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, int64_t groups,
                          ge::GeTensorDesc& newConvOutDesc, ge::NodePtr& concatNode, size_t dimIdx);

  bool GenerateNewNodes(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, ge::NodePtr& splitNode,
                        vector<ge::NodePtr>& newConvNodes, ge::NodePtr& concatNode, int64_t Conv2DGroups,
                        size_t dimChannelIdx, uint32_t anchorIdx);

  bool Relink(ge::NodePtr srcNode, ge::NodePtr splitNode, vector<ge::NodePtr>& newConvNodes, ge::NodePtr concatNode,
              int64_t Conv2DGroups, uint32_t anchorIdx);
  bool RemoveOldNode(ge::ComputeGraph& graph, ge::NodePtr srcNode);
  const string FUSED_OP_TYPE = "Conv2DBackpropInputD";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2DBACKPROP_GROUP_FUSION_PASS_H_
