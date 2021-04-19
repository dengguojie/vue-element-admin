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
* \file common_lstm_fusion_pass.h
* \brief COMMONLSTM fusion pass
* \brief COMMONLSTM fusion pass(LSTM-->DynamicRnn)
*/
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_COMMON_LSTM_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_COMMON_LSTM_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
// Common LSTM input index map
struct InputIndexInfo {
  int32_t wIndex = 1;
  int32_t rIndex = 2;
  int32_t biasIndex = 3;
};

class CommonLSTMFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;
 private:
  ge::GeTensorPtr ProcessLSTMWxh(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo, int32_t &hiddenSize);
  ge::GeTensorPtr ProcessLSTMBias(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo, int32_t hiddenSize);
  void SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims, const ge::Format &format,
                            const ge::DataType &dtype);
  Status AddReshapeNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode,
                        ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes, std::string nodeName,
                        int nodeIndex);
  const string FUSED_OP_TYPE = "CommonLSTM";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_COMMON_LSTM_FUSION_PASS_H