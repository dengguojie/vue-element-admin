/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief bnhost fusion pass
 *
 */

#ifndef BNHOST_FUSION_PASS_H
#define BNHOST_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class HostBNFusionPass: public PatternFusionBasePass {


 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
               vector<ge::NodePtr> &newNodes) override;

 private:


  /**
   * Do SwapCo fusion for PSROIPooling
   * @param graph: original graph info
   * @param convNodePtr: conv2d node info
   * @param psroiNodePtr: PSROIPooling node
   * @param newNodes: new nodes after fusion
   * @return SUCCESS/FAILED
   */
  Status BNFuison(ge::ComputeGraph &graph,
                      ge::NodePtr &bnNodePtr, vector<ge::NodePtr> &newNodes);

  /**
   * Check parameters of bn right or not
   * @param bnNodePtr: bn node
   * @return SUCCESS/FAILED
   */
  Status CheckParameter(ge::NodePtr &bnNodePtr);

  /**
   * Set output_dim and group_size attr value
   * @param newNodePtr: new node
   * @return SUCCESS/FAILED
   */
  Status SetAttrValueForNewNode(const ge::OpDescPtr &psroiOpDescPtr,
                                ge::OpDescPtr &newOpDescPtr);

  /**
   * Get new input desc info of SwapCo or SwapCi
   * @param currentOpDescPtr: current op desc(SwapCo/SwapCi)
   * @param preOpDescPtr: pre op desc
   * @param inputTensorDesc: old input desc of SwapCo or SwapCi
   * @return SUCCESS/FAILED
   */
  Status GetSwapInputTensorDesc(const ge::OpDescPtr &currentOpDescPtr,
                                const ge::OpDescPtr &preOpDescPtr,
                                ge::GeTensorDesc &inputTensorDesc);
  Status GetInputDataTensorDesc(const ge::NodePtr &dataNodePtr,
                                const ge::NodePtr &bnNodePtr,
                                ge::GeTensorDesc &inputTensorDesc);

  /**
   * Get new input desc info of SwapCi
   * @param currentOpDescPtr: current op desc(SwapCi)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCi
   * @param outputTensorDesc: new out desc of SwapCi
   * @return SUCCESS/FAILED
   */
  Status GetSwapCiOutputTensorDesc(const ge::OpDescPtr &currentOpDescPtr,
                                   const ge::OpDescPtr &nextOpDescPtr,
                                   const ge::GeTensorDesc &inputTensorDesc,
                                   ge::GeTensorDesc &outputTensorDesc);

  /**
   * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetMeanOutputTensorDesc(const ge::OpDescPtr &currentOpDescPtr,
                                   const ge::OpDescPtr &nextOpDescPtr,
                                   const ge::GeTensorDesc &inputTensorDesc,
                                   ge::GeTensorDesc &outputTensorDesc);

      /**
   * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetVarOutputTensorDesc(const ge::OpDescPtr &currentOpDescPtr,
                                   const ge::OpDescPtr &nextOpDescPtr,
                                   const ge::GeTensorDesc &inputTensorDesc,
                                   ge::GeTensorDesc &outputTensorDesc);
   /**
      * Get new output desc info of SwapCo
   * @param currentOpDescPtr: current op desc(SwapCo)
   * @param nextOpDescPtr: next op of PSROIPooling
   * @param inputTensorDesc: input desc of SwapCo
   * @param outputTensorDesc: new out desc of SwapCo
   * @return SUCCESS/FAILED
   */
  Status GetMuOutputTensorDesc(const ge::OpDescPtr &currentOpDescPtr,
                                   const ge::OpDescPtr &nextOpDescPtr,
                                   const ge::GeTensorDesc &inputTensorDesc,
                                   ge::GeTensorDesc &outputTensorDesc);

  Status GetInferOutputTensorDesc(
        const ge::OpDescPtr &hostOpDescPtr, const ge::OpDescPtr &currentOpDescPtr,
        const ge::GeTensorDesc &inputTensorDesc,
        ge::GeTensorDesc &outputTensorDesc);

  Status SetAttrValue(
        const ge::OpDescPtr &preOpDescPtr, ge::OpDescPtr &newOpDescPtr);

  const string FUSED_OP_TYPE = "BNInference";
};
} // namespace fe

#endif // BNHOST_FUSION_PASS_H
