/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file avg_pool_grad_fusion_pass.h
 * \brief avg_pool_grad fusion pass(avg_pool_grad --> avg_pool_grad_d)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_GRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_GRAD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AvgPoolGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status AvgValueTableGen(vector<int64_t> dimInfo, vector<int64_t> Ksize, vector<int64_t> strides, string padding,
                          string data_format, vector<int64_t>& assitDimInfo, uint16_t* output);
  Status WindowedOutputSize(int32_t input, int32_t kSize, int32_t stride, string padding, int32_t& output,
                            int32_t& padBefor, int32_t& padAfter);
  Status TransposeNCHW2NHWC(int32_t nOutput, int32_t hOutput, int32_t wOutput, int32_t cOutput, uint16_t* avgpoolout);

  const string FUSED_OP_TYPE = "AVGPOOLGRAD";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_GRAD_FUSION_PASS_H_