/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file avg_pool_1d_fusion_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_1D_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_1D_FUSION_PASS_H_

#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe {
class AvgPool1DFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  template <typename T>
  Status AvgValueTableGen(const vector<int64_t>& dim_info, const int64_t kernel_size, const int64_t stride_size,
                          const vector<int64_t>& padding, const bool ceil_mode, const bool count_include_pad,
                          vector<int64_t>& assit_dim_info, T* output);
  const string kFusedOpType = "AvgPool1DD";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_1D_FUSION_PASS_H_