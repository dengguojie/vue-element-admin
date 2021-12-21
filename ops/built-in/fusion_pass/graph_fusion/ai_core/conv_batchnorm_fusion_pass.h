/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file conv_batchnorm_fusion_pass.h
 * \brief fuse conv batchnorm
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_BATCHNORM_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_BATCHNORM_FUSION_PASS_H_

#include <vector>
#include "conv_fusion_pass_base.h"

namespace fe {
class ConvBatchnormFusionPass : public ConvFusionPassBase {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status CheckWeights(const ge::NodePtr bnNode);
  bool IsBatchNormMultiOutput(ge::NodePtr &destNode);

  const size_t BN_INFERENCE_D_WEIGHT_SIZE = 2;
  const size_t BATCHNORM_MAXIMUM_WEIGHT_SIZE = 4;

  const char* BN_INFERENCE_D = "BNInference";
  const char* BATCHNORM = "BatchNorm";
  const char* STREAMSWITCH = "StreamSwitch";

  const char* NEED_ADD_AND_RSQRT = "need_adding_eps_and_rsqrting";
  const char* KERNEL_NUM = "kernelNum";
  const char* HAS_BIAS = "hasbias";

  const float EPS = 1e-8;
  const float EPS_DEFAULT_FLOAT = 1e-5;

  const string FUSED_OP_TYPE = "Conv2D/Conv3D/DepthwiseConv2D";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_BATCHNORM_FUSION_PASS_H_
