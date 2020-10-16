/**
 * @file conv2d_bp_input_elemwise_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d_backprop_input + elemwise ops fusion pattern
 *
 * @version 1.0
 *
 */


#ifndef TBE_DX_ELEMWISE_PASS_H
#define TBE_DX_ELEMWISE_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeDxElemwisePass : public BufferFusionPassBase {
    public:
        explicit TbeDxElemwisePass() {}

        ~TbeDxElemwisePass() {}

    protected:
        vector<BufferFusionPattern *> DefinePatterns() override;
        Status GetFusionNodes(const BufferFusionMapping &mapping,
                              vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "FusedOp";
};

}  // namespace fe

#endif  // TBE_DX_ELEMWISE_PASS_H
