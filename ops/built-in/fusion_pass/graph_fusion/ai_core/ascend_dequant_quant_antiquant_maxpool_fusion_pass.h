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
 * \file ascend_dequant_quant_antiquant_maxpool_fusion_pass.h
 * \brief ascend_dequant_quant_antiquant_maxpool fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ASCEND_DEQUANT_QUANT_ANTIQUANT_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ASCEND_DEQUANT_QUANT_ANTIQUANT_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <string>
#include <vector>

namespace fe {
class AscendDequantQuantAntiquantMaxpoolFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status CheckPeerAllInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor, const size_t& expectedNum);
  Status IsMatch(ge::NodePtr& deqNode, ge::NodePtr& quantNode, ge::NodePtr& antiqNode, ge::NodePtr& maxpoolNode);
  const string FUSED_OP_TYPE = "AscendDeqQuantAntiq";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ASCEND_DEQUANT_QUANT_ANTIQUANT_FUSION_PASS_H_
