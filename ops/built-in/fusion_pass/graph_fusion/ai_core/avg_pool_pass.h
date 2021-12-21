/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file avg_pool_pass.h
 * \brief avg_pool fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AvgPoolFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mulNode, string& padding, vector<int64_t>& dimInfo,
                  vector<int64_t> ksize, vector<int64_t> stride);
  ge::NodePtr AddMul(ge::ComputeGraph& graph, ge::NodePtr& avgPoolNode, ge::Format& inputOriginFormat);
  Status Calc4DWeightAvgPool(const std::vector<int64_t>& filterDims4D, const int64_t& kernelDataCount,
                             const int8_t* filterInt8Data, std::unique_ptr<int32_t[]>& weightInt8Temp);
  Status DoBiasOptimizeAvgpool(ge::ComputeGraph& graph, ge::NodePtr poolingNode, vector<ge::NodePtr>& fusionNodes,
		               int64_t& ksizeH, int64_t& ksizeW, int64_t& inputC);
  Status GetWeightOfConvAvgpool(const std::string& opName, const int8_t* filterInt8Data,
                                const std::vector<int64_t>& filterDims, std::unique_ptr<int32_t[]>& weightInt8OutParam);
  Status UpdateDequantConst(ge::ComputeGraph& graph, ge::NodePtr& const_node, float& area_factor);
  Status SetInvalidKernelAvgpool(ge::ComputeGraph& graph, ge::NodePtr fusedNode);
  const string FUSED_OP_TYPE = "AvgPool";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_AVG_POOL_PASS_H_
