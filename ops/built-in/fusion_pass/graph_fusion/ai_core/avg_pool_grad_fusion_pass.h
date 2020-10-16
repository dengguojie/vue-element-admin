/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   avg_pool_grad_fusion_pass.h
 *
 * @brief  avg_pool_grad fusion pass(avg_pool_grad --> avg_pool_grad_d)
 *
 */

#ifndef FE_AVG_POOL_GRAD_FUSION_H
#define FE_AVG_POOL_GRAD_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    class AvgPoolGradFusionPass : public PatternFusionBasePass {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
      Status Fusion(ge::ComputeGraph &graph,
                    Mapping &mapping,
                    vector<ge::NodePtr> &fusionNodes) override;

    private:
        Status AvgValueTableGen(vector<int64_t> dimInfo, vector<int64_t> Ksize, vector<int64_t> strides, string padding,
                                string data_format, vector<int64_t> &assitDimInfo, uint16_t *output);
        Status WindowedOutputSize(int32_t input, int32_t kSize, int32_t stride, string padding, int32_t &output,
                                  int32_t &padBefor, int32_t &padAfter);
        Status TransposeNCHW2NHWC(int32_t nOutput, int32_t hOutput, int32_t wOutput, int32_t cOutput, uint16_t* avgpoolout);

        const string FUSED_OP_TYPE = "AVGPOOLGRAD";
    };

}  // namespace fe

#endif  // FE_FE_AVG_POOL_GRAD_FUSION_H_FUSION_H