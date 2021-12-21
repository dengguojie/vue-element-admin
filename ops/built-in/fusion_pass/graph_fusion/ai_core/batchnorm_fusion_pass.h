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
 * \file batchnorm_fusion_pass.h
 * \brief Fused Add, Mul(three), Sub of structure:
 *           const     const
 *               \    /
 *                Mul  const
 *              /   \  /
 *  Conv3d    /     Mul  const
 *      \   /        |  /
 *       Mul        Sub
 *         \       /
 *          \    /
 *           Add
 *
 *          or :
 *             const(variance)  const(eps)
 *                      \     /
 *                       Add
 *                        |
 *                      Rsqrt
 *                       |
 *          const       /
 *               \    /
 *                Mul  const
 *              /   \  /
 *  Conv3d    /     Mul  const
 *      \   /        |  /
 *       Mul        Sub
 *         \       /
 *          \    /
 *           Add
 * into batch norm op fusion pass
 *
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCHNORM_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCHNORM_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchnormFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status AddTensorDescForBn(const ge::OpDescPtr& bnOpdesc, const ge::GeTensorDesc& inputTensor,
                            const ge::GeTensorDesc& scaleTensor, const ge::GeTensorDesc& offsetTensor,
                            const ge::GeTensorDesc& meanTensor, const ge::GeTensorDesc& varianceTensor,
                            const ge::GeTensorDesc& bnOutTensor);
  Status CheckInputTensorValid(const ge::GeTensorDesc& tensorDesc, const int64_t& kernelNum);
  Status CheckInputTypeValid(const ge::NodePtr& originalNode, const ge::NodePtr& inputNode, const string& expectOpType);
  Status CheckPeerInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor, const size_t& expectedNum);
  int64_t GetKernelNumOfOutputOfConv3D(const ge::NodePtr& conv);
  Status RemoveSmalleNodes(ge::ComputeGraph& graph, const ge::NodePtr& addNode, const ge::NodePtr& mulNode1,
                           const ge::NodePtr& mulNode2, const ge::NodePtr& mulNode3, const ge::NodePtr& subNode);
  const string FUSED_OP_TYPE = "BatchNorm";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCHNORM_FUSION_PASS_H_
