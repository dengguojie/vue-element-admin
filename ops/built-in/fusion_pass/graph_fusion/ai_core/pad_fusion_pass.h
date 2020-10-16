/**
 * @file pad_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(pad --> pad_d)
 *
 */

#ifndef FE_PAD_FUSION_H
#define FE_PAD_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
class PadFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    bool GetConstValue(const ge::Operator &op, const ge::Tensor &const_tensor, const ge::DataType &dtype,
                                      std::vector<int64_t> &const_data);
    Status PadMoveConsttoAttr(ge::ComputeGraph &graph, ge::NodePtr &pad_node, const string &attr_name, int32_t index);
    const string FUSED_OP_TYPE = "PadD";
};

}  // namespace fe

#endif  // FE_PAD_FUSION_H
