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
 * \file pooling_matrix_pass.h
 * \brief Pooling fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_POOLING_MATRIX_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_POOLING_MATRIX_PASS_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class PoolingFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mulNode, const vector<int64_t> pad, const vector<int64_t>& dimInfo,
                  const vector<int64_t> window, const vector<int64_t> stride, std::string& recode, const bool isInt8);
  ge::NodePtr AddMul(ge::ComputeGraph& graph, const ge::NodePtr& avgPoolNode);
  Status Calc4DWeight(const std::vector<int64_t>& filterDims4D, const int64_t& kernelDataCount,
                      const int8_t* filterInt8Data, std::unique_ptr<int32_t[]>& weightInt8Temp);
  Status DoBiasOptimize(ge::ComputeGraph& graph, ge::NodePtr poolingNode, vector<ge::NodePtr>& fusionNodes,
                        const int64_t& windowH, const int64_t& windowW, const int64_t& inputC);
  Status GetWeightOfConv(const std::string& opName, const int8_t* filterInt8Data,
                         const std::vector<int64_t>& filterDims, std::unique_ptr<int32_t[]>& weightInt8OutParam);
  bool IsMeanValueAllEqual(vector<int64_t> input, vector<int64_t> window, vector<int64_t> stride, vector<int64_t> pad,
                           int64_t ceil_mode);
  const string FUSED_OP_TYPE = "Pooling";
  unordered_map<string, ge::NodePtr> mulConstNode;
  unordered_map<string, ge::NodePtr> poolConstNode;
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_POOLING_MATRIX_PASS_H_
