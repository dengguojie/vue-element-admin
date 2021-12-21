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
 * \file hostin_fusion_pass.h
 * \brief bnhost fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTIN_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTIN_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class HostINFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;

  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  /**
   * Do SwapCo fusion for PSROIPooling
   * @param graph: original graph info
   * @param convNodePtr: instance norm node info
   * @param newNodes: new nodes after fusion
   * @return SUCCESS/FAILED
   */
  Status INFuison(ge::ComputeGraph& graph, ge::NodePtr& inNodePtr, vector<ge::NodePtr>& newNodes);

  /**
   * Check parameters of bn right or not
   * @param inNodePtr: bn node
   * @return SUCCESS/FAILED
   */
  Status CheckParameter(ge::NodePtr& inNodePtr);

  /**
   * Set output_dim and group_size attr value
   * @param newNodePtr: new node
   * @return SUCCESS/FAILED
   */
  Status SetAttrValueForNewNode(const ge::OpDescPtr& psroiOpDescPtr, ge::OpDescPtr& newOpDescPtr);

  const string FUSED_OP_TYPE = "INInferV2D";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTIN_FUSION_PASS_H_
