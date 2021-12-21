/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
* \file dynamic_gru_v2_transdatarnn_fusion_pass.h
* \brief DynamicGRUV2Trans fusion pass
* \brief DynamicGRUV2
*/
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_TRANSDATARNN_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_TRANSDATARNN_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicGRUV2TransFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;
 private:
  const string FUSED_OP_TYPE = "DynamicGRUV2Trans";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_TRANSDATARNN_FUSION_PASS_H