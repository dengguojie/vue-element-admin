/**
 * @file pad_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(pad --> pad_d)
 *
 */

#ifndef FE_PAD_V3_FUSION_H
#define FE_PAD_V3_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
class PadV3FusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool GetConstValue(const ge::Tensor &const_tensor, const ge::DataType &dtype,
                     std::vector<int64_t> &const_data);
  bool AutoRemoveInput(ge::ComputeGraph &graph, ge::NodePtr &pad_node, ge::Operator &op,
                       const string input_name);
  Status PadMoveConsttoAttr(ge::ComputeGraph& graph, ge::NodePtr& pad_node);
  const string FUSED_OP_TYPE = "PadV3D";
};

}  // namespace fe

#endif  // FE_PAD_V3_FUSION_H
