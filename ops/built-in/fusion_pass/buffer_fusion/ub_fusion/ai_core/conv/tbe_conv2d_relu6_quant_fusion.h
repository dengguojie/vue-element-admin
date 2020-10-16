/**
 * @file tbe_conv2d_relu6_quant_fusion.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe convolution + relu6 + quant ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef TBE_CONV2D_RELU6_QUANT_FUSION_PASS_H
#define TBE_CONV2D_RELU6_QUANT_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeConv2dRelu6QuantFusion : public BufferFusionPassBase {
    public:
        explicit TbeConv2dRelu6QuantFusion() {}

        ~TbeConv2dRelu6QuantFusion() {}

    protected:
        vector<BufferFusionPattern *> DefinePatterns() override;
        Status GetFusionNodes(const BufferFusionMapping &mapping,
                              vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // TBE_CONV2D_RELU6_QUANT_FUSION_PASS_H