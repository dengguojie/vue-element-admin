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
 * \file dynamic_rnn_v3_fusion_pass.h
 * \brief DynamicRNNV3 fusion pass
 * \brief DynamicRNNV3 fusion pass(DynamicRnnV3)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_V3_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_V3_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

// Caffe LSTM input index map

class DynamicRNNV3FusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;

private:
    ge::GeTensorPtr ProcessDynamicRnnV3Wdate(ge::NodePtr fusedNode, int64_t index, int64_t batchSize,
                                             int64_t hiddenSize);
    ge::NodePtr AddBroadCastForCt(ge::ComputeGraph &graph, ge::NodePtr fusedNode, bool &failStatus,
                                  int64_t batchSize, int64_t stateSize);
    const string FUSED_OP_TYPE = "DynamicRNNV3";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_V3_FUSION_PASS_H
