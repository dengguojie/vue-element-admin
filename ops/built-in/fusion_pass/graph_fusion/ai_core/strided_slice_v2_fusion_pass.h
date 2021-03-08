/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(strided_slice_v2--> strided_slice_v2_d)
 *
 */

#ifndef FE_STRIDEDSLICEV2_FUSION_H
#define FE_STRIDEDSLICEV2_FUSION_H

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"

namespace fe {
class ConstToAttrStridedSliceV2Pass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
  Status GetReverseState(const Operator &op, ge::NodePtr &fused_node,
                         ge::OpDescPtr fuse_desc,
                         std::vector<int64_t> &new_axes, bool &need_to_reverse);
  Status GetStridedSliceV2CpuState(const Operator &op, ge::NodePtr &fused_node,
                                   ge::OpDescPtr fuse_desc, bool &need_to_cpu);
  bool ApplyAxesToAttr(const ge::Operator op, std::vector<int64_t> &new_begins,
                       std::vector<int64_t> &new_ends,
                       std::vector<int64_t> &new_axes,
                       std::vector<int64_t> &new_strides) const;
  bool CheckMask(const int64_t new_mask, const int64_t shrink_mask,
                 const size_t dim_num) const;
  bool AutoRemoveInput(ge::ComputeGraph &graph, ge::NodePtr &p_node,
                       const int64_t index);
  bool GetConstValue(const Operator &op, const Tensor &const_tensor,
                     const DataType &dtype, std::vector<int64_t> &const_data);
  void UpdateShapeAndDataType(ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc);
  void MakeConstNode(ge::NodePtr &fuse_node, ge::OpDescPtr fuse_desc);
  Status SetConstDesc(vector<int64_t> &tensor_shape,
                      ge::GeTensorDesc &tensor_desc,
                      ge::GeTensorDesc &des_desc) const;

 private:
  static const int64_t DEFAULT_STRIDE;
  static const std::string FUSEDNODE;
  static const std::string PATTERN_FUSEDNODE;
};
}  // namespace fe

#endif  // FE_STRIDEDSLICEV2_FUSION_H
