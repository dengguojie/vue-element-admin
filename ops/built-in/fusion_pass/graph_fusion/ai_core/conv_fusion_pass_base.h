/**
 * @file conv_scale_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fuse conv and other op
 *
 * @version 1.0
 *
 */
#ifndef _FE_CONV_FUSION_BASE_H_
#define _FE_CONV_FUSION_BASE_H_

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
const int32_t DHWCN_DIM_N = 4;
class ConvFusionPassBase : public PatternFusionBasePass {
 public:
  Status DoFusion(ge::ComputeGraph &graph, ge::NodePtr convNode,
                  ge::NodePtr destNode, vector<ge::NodePtr> &fusionNodes);

  Status GetConvFilterInputIndex(const ge::NodePtr &convNode,
                                 int &filterInputIdx);

  Status GetAllConstInput(const ge::NodePtr &node,
                          vector<ge::GeTensorDesc> &conv2dInputs,
                          vector<string> &conv2dInputsName,
                          vector<ge::InDataAnchorPtr> &conv2dInputAncors,
                          vector<ge::GeTensorDesc> &constOutputs,
                          vector<ge::OutDataAnchorPtr> &constOutputAncors);

  Status AddBiasNode(ge::ComputeGraph &graph, ge::NodePtr &convNode);

  Status GetConvKernelIndex(ge::OpDescPtr convOpdesc,
                            const ge::GeTensorDesc &constInputDesc,
                            ge::Format &filterFormat, size_t &kernerlIndex);
  const float FLOAT_NUM_ZERO = 0.;
};
}  // namespace fe
#endif  // _FE_CONV_SCALE_FUSION_H_
