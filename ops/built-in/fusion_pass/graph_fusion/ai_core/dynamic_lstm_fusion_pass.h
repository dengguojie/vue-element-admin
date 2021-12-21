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
 * \file dynamic_lstm_fusion_pass.h
 * \brief DynamicLSTM fusion pass
 * \brief DynamicLSTM fusion pass(LSTM --> DynamicLSTM & FullyConnection)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_LSTM_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_LSTM_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

// Caffe LSTM input index map
struct InputIndexInfo {
    int32_t xStaticIndex = -1;
    int32_t h0Index = -1;
    int32_t c0Index = -1;
    int32_t wxIndex = -1;
    int32_t biasIndex = -1;
    int32_t whIndex = -1;
    int32_t wxStaticIndex = -1;
    bool    hasStatic = false;
};

class DynamicLSTMFusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;

private:
    Status ProcessLSTMStatic(ge::NodePtr fusedNode, ge::NodePtr &innerproductNode, ge::ComputeGraph &graph, 
        vector<ge::NodePtr> &newNodes, const InputIndexInfo &inputIndexInfo);
    ge::GeTensorPtr ProcessLSTMWxh(ge::NodePtr fusedNode, bool &failStatus, const InputIndexInfo &inputIndexInfo);
    Status AddDynamicLSTMNode(ge::OpDescPtr &thisOpDesc, const ge::OpDescPtr &formerOpdesc,
        const ge::GeTensorDesc &wxhTensorDesc, const InputIndexInfo &inputIndexInfo, bool expose_hidden, ge::GeTensorDesc &staticTensorDesc,
        int32_t outputSize);
    const string FUSED_OP_TYPE = "DynamicLSTMV2";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_LSTM_FUSION_PASS_H
