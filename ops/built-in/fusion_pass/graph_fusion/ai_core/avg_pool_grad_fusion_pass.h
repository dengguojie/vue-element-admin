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
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  Status AvgValueTableGen(const vector<int64_t> dim_info, const vector<int64_t> k_size, const vector<int64_t> strides,
                          const string padding, const string data_format, vector<int64_t>& assit_dim_info,
                          uint16_t* output);
  Status WindowedOutputSize(const int32_t input, const int32_t k_size, const int32_t stride, const string padding,
                            int32_t& output, int32_t& pad_befor, int32_t& pad_after);
  Status TransposeNCHW2NHWC(const int32_t n_output, const int32_t h_output, const int32_t w_output,
                            const int32_t c_output, uint16_t* avgpoolout);

  const string kFusedOpType = "AVGPOOLGRAD";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_GRAD_FUSION_PASS_H_