/**
 * @file conv_scale_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fuse conv scale
 *
 * @version 1.0
 *
 */
#ifndef _FE_CONV_SCALE_FUSION_H_
#define _FE_CONV_SCALE_FUSION_H_

#include <vector>
#include "conv_fusion_pass_base.h"

namespace fe {
class ConvScaleFusionPass : public ConvFusionPassBase {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  const char *SCALE = "Scale";
  const char *KERNEL_NUM = "kernelNum";
  const char *HAS_BIAS = "hasbias";
  const string FUSED_OP_TYPE = "Conv2D/DepthwiseConv2D";
};
}  // namespace fe
#endif  // _FE_CONV_SCALE_FUSION_H_
