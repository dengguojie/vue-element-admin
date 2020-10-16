/**
 * @file conv2d_writeselect_stridewrite_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief tbe conv2d + write_select + stride_write ops fusion pattern
 *
 * @version 1.0
 *
 */

#ifndef TBE_CONV2D_WRITESELECT_STRIDEDWRITE_FUSION_PASS_H
#define TBE_CONV2D_WRITESELECT_STRIDEDWRITE_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {

class TbeConv2dWrtselStridewrtPass : public BufferFusionPassBase {
    public:
        explicit TbeConv2dWrtselStridewrtPass() {}

        ~TbeConv2dWrtselStridewrtPass() {}

    protected:
        vector<BufferFusionPattern *> DefinePatterns() override;

private:
    const string FUSED_OP_TYPE = "FusedOp";
};
}  // namespace fe

#endif  // TBE_CONV2D_WRITESELECT_STRIDEDWRITE_FUSION_PASS_H