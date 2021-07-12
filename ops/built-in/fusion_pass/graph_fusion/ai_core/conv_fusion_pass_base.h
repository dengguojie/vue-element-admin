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
 * \file conv_fusion_pass_base.h
 * \brief fuse conv and other op
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_BASE_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_BASE_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
static const string CONV2D = "Conv2D";
static const std::string CONSTANT = "Const";
const std::string CONSTANTOP = "Constant";
const std::string CONVBNFILTERHOST = "ConvBnFilterHost";
const std::string CONVBNBIASHOST = "ConvBnBiasHost";
const std::string DEPTHWISECONV2D = "DepthwiseConv2D";
const std::string CONV3D = "Conv3D";
const std::string GROUPPADDING = "GroupPadding";
const std::string CONCATHOSTOP = "Concatv2HostCpuOp";
const std::string QUANTWEIGHTROLLBACK = "QuantWeightRollBack";
static const std::string IS_DEPTHWISE_CONV2D = "_is_depthwise_conv2d";
const int32_t NCHW_DIM_N = 0;
const int32_t NCHW_DIM_C = 1;
const int32_t NCHW_DIM_H = 2;
const int32_t NCHW_DIM_W = 3;
const int32_t NHWC_DIM_N = 0;
const int32_t NHWC_DIM_H = 1;
const int32_t NHWC_DIM_W = 2;
const int32_t NHWC_DIM_C = 3;
const int32_t HWCN_DIM_H = 0;
const int32_t HWCN_DIM_W = 1;
const int32_t HWCN_DIM_C = 2;
const int32_t HWCN_DIM_N = 3;
const int32_t CHWN_DIM_C = 0;
const int32_t CHWN_DIM_H = 1;
const int32_t CHWN_DIM_W = 2;
const int32_t CHWN_DIM_N = 3;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;
class ConvFusionPassBase : public PatternFusionBasePass {
 public:
  Status DoFusion(ge::ComputeGraph& graph, ge::NodePtr convNode, ge::NodePtr destNode,
                  vector<ge::NodePtr>& fusionNodes);

  Status GetConvFilterInputIndex(const ge::NodePtr& convNode, int& filterInputIdx);

  Status GetAllConstInput(const ge::NodePtr& node, vector<ge::GeTensorDesc>& conv2dInputs,
                          vector<string>& conv2dInputsName, vector<ge::InDataAnchorPtr>& conv2dInputAncors,
                          vector<ge::GeTensorDesc>& constOutputs, vector<ge::OutDataAnchorPtr>& constOutputAncors);

  Status AddBiasNode(ge::ComputeGraph& graph, ge::NodePtr& convNode);

  Status GetConvKernelIndex(ge::OpDescPtr convOpdesc, const ge::GeTensorDesc& constInputDesc, ge::Format& filterFormat,
                            size_t& kernerlIndex);
  Status GetConvChannelIndex(ge::OpDescPtr convOpdesc, const ge::GeTensorDesc& constInputDesc, ge::Format& filterFormat,
                            size_t& channelIndex);
  const float FLOAT_NUM_ZERO = 0.;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_BASE_H_
