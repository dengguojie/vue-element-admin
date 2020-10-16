/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   avg_pool_v2_grad_fusion_pass.h
 *
 * @brief  avg_pool_v2_grad fusion pass(avg_pool_v2_grad --> avg_pool_v2_grad_d)
 *
 */

#ifndef FE_AVG_POOL_V2_GRAD_FUSION_H
#define FE_AVG_POOL_V2_GRAD_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    class AvgPoolV2GradFusionPass : public PatternFusionBasePass {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
      Status Fusion(ge::ComputeGraph &graph,
                    Mapping &mapping,
                    vector<ge::NodePtr> &fusionNodes) override;
    private:
        Status AvgValueTableGenV2(vector<int64_t> dimInfo, vector<int64_t> Ksize, vector<int64_t> strides, string padding,
                                  vector<int64_t> pads, string data_format, bool ceil_mode, vector<int64_t> &assitDimInfo,
                                  uint16_t *output);
        Status WindowedOutputSizeV2(int32_t input, int32_t kSize, int32_t stride, string padding, int32_t &output,
                                    int32_t pad_1, int32_t pad_2, int32_t &padBefor, int32_t &padAfter, bool ceil_mode);
        Status TransposeNCHW2NHWCV2(int32_t nOutput, int32_t hOutput, int32_t wOutput, int32_t cOutput, uint16_t* avgpoolout);
        Status KernelGenV2(int32_t hKsize, int32_t wKsize, int32_t cInput, vector<int64_t> assitDimInfo, uint16_t *kernelTable);
        const string FUSED_OP_TYPE = "AVGPOOLV2GRAD";
    };

}  // namespace fe

#endif  // FE_AVG_POOL_V2_GRAD_FUSION_H