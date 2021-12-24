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
 * \file hostbn_fusion_pass.h
 * \brief bnhost fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTBN_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTBN_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class HostBNFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  /**
   * Do SwapCo fusion for PSROIPooling
   * @param graph: original graph info
   * @param convNodePtr: conv2d node info
   * @param psroiNodePtr: PSROIPooling node
   * @param newNodes: new nodes after fusion
   * @return SUCCESS/FAILED
   */
  Status BNFuison(ge::ComputeGraph& graph, ge::NodePtr& bnNodePtr, vector<ge::NodePtr>& newNodes);

  /**
   * Check parameters of bn right or not
   * @param bnNodePtr: bn node
   * @return SUCCESS/FAILED
   */
  Status CheckParameter(const ge::NodePtr& bnNodePtr) const;

  /**
   * Set output_dim and group_size attr value
   * @param newNodePtr: new node
   * @return SUCCESS/FAILED
   */
  Status SetAttrValueForNewNode(const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr) const;

  /**
   * Get new input desc info of SwapCo or SwapCi
   * @param currentOpDescPtr: current op desc(SwapCo/SwapCi)
   * @param preOpDescPtr: pre op desc
   * @param inputTensorDesc: old input desc of SwapCo or SwapCi
   * @return SUCCESS/FAILED
   */
  Status GetSwapInputTensorDesc(const ge::OpDescPtr& preOpDescPtr,
                                ge::GeTensorDesc& inputTensorDesc) const;
  Status GetInputDataTensorDesc(const ge::NodePtr& dataNodePtr, const ge::NodePtr& preNodePtr,
                                ge::GeTensorDesc& inputTensorDesc) const;

  /**
   * Get new input desc info of SwapCi
   * @param currentOpDescPtr: current op desc(SwapCi)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCi
   * @param outputTensorDesc: new out desc of SwapCi
   * @return SUCCESS/FAILED
   */
  Status GetSwapCiOutputTensorDesc(const ge::OpDescPtr& currentOpDescPtr, const ge::OpDescPtr& nextOpDescPtr,
                                   const ge::GeTensorDesc& inputTensorDesc, ge::GeTensorDesc& outputTensorDesc);

  /**
   * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetMeanOutputTensorDesc(const ge::OpDescPtr& currentOpDescPtr,
                                 const ge::GeTensorDesc& inputTensorDesc, ge::GeTensorDesc& outputTensorDesc) const;

  /**
   * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetVarOutputTensorDesc(const ge::OpDescPtr& currentOpDescPtr,
                                const ge::GeTensorDesc& inputTensorDesc, ge::GeTensorDesc& outputTensorDesc) const;
  /**
   * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetMuOutputTensorDesc(const ge::OpDescPtr& currentOpDescPtr,
                               const ge::GeTensorDesc& inputTensorDesc, ge::GeTensorDesc& outputTensorDesc) const;

  Status GetInferOutputTensorDesc(const ge::OpDescPtr& currentOpDescPtr,
                                  ge::GeTensorDesc& outputTensorDesc) const;

  Status SetAttrValue(const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr) const;

  const string FUSED_OP_TYPE = "BNInference";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_HOSTBN_FUSION_PASS_H_
