/**
 * @file conv_batchnorm_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fuse conv batchnorm
 *
 * @version 1.0
 *
 */
#ifndef _FE_CONV_BATCHNORM_FUSION_H_
#define _FE_CONV_BATCHNORM_FUSION_H_

#include <vector>
#include "conv_fusion_pass_base.h"

namespace fe {
class ConvBatchnormFusionPass : public ConvFusionPassBase {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  Status CheckWeights(const ge::NodePtr bnNode);

  const size_t BN_INFERENCE_D_WEIGHT_SIZE = 2;
  const size_t BATCHNORM_MAXIMUM_WEIGHT_SIZE = 4;

  const char *BN_INFERENCE_D = "BNInference";
  const char *BATCHNORM = "BatchNorm";
  const char *STREAMSWITCH = "StreamSwitch";

  const char *NEED_ADD_AND_RSQRT = "need_adding_eps_and_rsqrting";
  const char *KERNEL_NUM = "kernelNum";
  const char *HAS_BIAS = "hasbias";


  const float EPS = 1e-8;
  const float EPS_DEFAULT_FLOAT = 1e-5;

  const string FUSED_OP_TYPE = "Conv2D/Conv3D/DepthwiseConv2D";
};
}  // namespace fe
#endif  // _FE_CONV_BATCHNORM_FUSION_H_
