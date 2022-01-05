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
  int32_t inithIndex = 5;
  int32_t initcIndex = 6;
};

class CommonLSTMFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;
 private:
  Status ProcessLSTMWxh(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo, int32_t &hiddenSize,
                        int32_t num_directions, vector<ge::GeTensorPtr> &tensorPtr);
  Status ProcessLSTMBias(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                         int32_t num_directions, int32_t hiddenSize, vector<ge::GeTensorPtr> &tensorPtr);
  void SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims, const ge::Format &format,
                            const ge::DataType &dtype);
  Status AddReshapeNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode,
                        ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes, std::string nodeName,
                        int nodeIndex);
  Status AddExpandDimsNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode,
                           ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes, std::string nodeName,
                           int nodeIndex);
  Status AddSliceNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode,
                      ge::NodePtr dynamicRnnNode, ge::GeTensorDesc dynamicRnnOutputDesc,
                      vector<ge::NodePtr> &newNodes, const std::string& nodeName, int nodeIndex);
  Status AddRNNMaskNode(ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode, ge::ComputeGraph &graph,
                        int32_t hiddenSize, vector<ge::NodePtr> &newNodes);
  Status AddSliceConcatNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnForwardNode,
                            ge::NodePtr dynamicRnnReverseNode, ge::GeTensorDesc dynamicRnnOutputDesc,
                            vector<ge::NodePtr> &newNodes, std::string nodeName, int nodeIndex);
  ge::OpDescPtr CreateSplitDesc(ge::OpDescPtr splitDesc, ge::OpDescPtr fusedDesc,
                                string tensorName, int64_t splitDim);
  Status SetOutputTensorDescAttr(uint16_t originOutputIndex, uint16_t fuseOutputIndex,
                                 ge::NodePtr originNode, ge::NodePtr fuseNode);
  Status ProcessLSTMInitH(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                          vector<ge::GeTensorPtr> &tensorPtr);
  Status ProcessLSTMInitC(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                          vector<ge::GeTensorPtr> &tensorPtr);
  const string FUSED_OP_TYPE = "CommonLSTM";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_COMMON_LSTM_FUSION_PASS_H