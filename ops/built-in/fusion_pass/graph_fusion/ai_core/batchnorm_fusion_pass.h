/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Fused Add, Mul(three), Sub of structure:
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
#ifndef FE_BATCHNORM_FUSION_PASS_H
#define FE_BATCHNORM_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchnormFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

 private:
    Status AddTensorDescForBn(const ge::OpDescPtr& bnOpdesc,
                                                   const ge::GeTensorDesc& inputTensor,
                                                   const ge::GeTensorDesc& scaleTensor,
                                                   const ge::GeTensorDesc& offsetTensor,
                                                   const ge::GeTensorDesc& meanTensor,
                                                   const ge::GeTensorDesc& varianceTensor,
                                                   const ge::GeTensorDesc& bnOutTensor);
    Status CheckInputTensorValid(const ge::GeTensorDesc& tensorDesc,
                                                      const int64_t& kernelNum);
    Status CheckInputTypeValid(const ge::NodePtr& originalNode,
                                                    const ge::NodePtr& inputNode,
                                                    const string& expectOpType);
    Status CheckPeerInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
            const size_t& expectedNum);
    int64_t GetKernelNumOfOutputOfConv3D(const ge::NodePtr& conv);
    Status RemoveSmalleNodes(ge::ComputeGraph &graph,
            const ge::NodePtr& addNode, const ge::NodePtr& mulNode1,
            const ge::NodePtr& mulNode2, const ge::NodePtr& mulNode3,
            const ge::NodePtr& subNode);
    const string FUSED_OP_TYPE = "BatchNorm";
};

}  // namespace fe

#endif  // FE_BATCHNORM_FUSION_PASS_H
