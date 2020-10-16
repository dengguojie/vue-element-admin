/**
 * @file conv2d_dequant_add_mul_quant_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + add + mul + quant ops fusion pattern
 *
 * @version 1.0
 *
 */


#ifndef TBE_CONV2D_ADD_MUL_QUNAT_PASS_H
#define TBE_CONV2D_ADD_MUL_QUNAT_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeConv2DAddMulQuantPass : public BufferFusionPassBase {
    public:
        explicit TbeConv2DAddMulQuantPass() {}

        ~TbeConv2DAddMulQuantPass() {}

    protected:
        vector<BufferFusionPattern *> DefinePatterns() override;
        Status GetFusionNodes(const BufferFusionMapping &mapping,
                              vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // TBE_CONV2D_ADD_MUL_QUNAT_PASS_H
