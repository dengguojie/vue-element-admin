/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file einsum_fusion_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EINSUM_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EINSUM_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class EinsumPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) override;

 private:
  Status CheckProduct(const std::vector<int64_t> &shape);
  Status CheckInputArgs(const Mapping &mapping);

  // ============= begin to define static shape handle =============
  // 001:reshape+reshape+matmul+reshape
  Status HandleStaticABCxCDE2ABDE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 008:reshape+reshape+matmul+reshape
  Status HandleStaticABCxDEC2ABDE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 009:reshape+reshape+matmul+reshape(swap input)
  Status HandleStaticABCxABDE2DEC(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 013:reshape+reshape+matmul+reshape(swap input)
  Status HandleStaticABCDxABE2ECD(ge::ComputeGraph &graph, ge::NodePtr &node);

  // ============= begin to define dynamic shape handle =============
  // 001:reshape+reshape+matmul+reshape
  Status HandleDynamicABCxCDE2ABDE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 008:reshape+reshape+matmul+reshape
  Status HandleDynamicABCxDEC2ABDE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 009:reshape+reshape+matmul+reshape(swap input)
  Status HandleDynamicABCxABDE2DEC(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 013:reshape+reshape+matmul+reshape(swap input)
  Status HandleDynamicABCDxABE2ECD(ge::ComputeGraph &graph, ge::NodePtr &node);

  // ============= begin to define common handle =============
  // 002 transpose+transpose+batchmatmul(swap input)
  Status HandleABCDxAECD2ACEB(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 003:transpose+batchmatmul+transpose
  Status HandleABCDxADBE2ACBE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 004:reshape+reshape+matmul+reshape-->reshape+batchmatmul
  Status HandleABCDxCDE2ABE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 005:reshape+matmul+reshape-->batchmatmul
  Status HandleABCxCD2ABD(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 006:reshape+matmul+reshape-->batchmatmul
  Status HandleABCxDC2ABD(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 007:reshape+reshape+matmul(swap input)
  Status HandleABCxABD2DC(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 010:transpose+batchmatmul+transpose
  Status HandleABCDxAECD2ACBE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 011:transpose+batchmatmul+transpose(swap input)
  Status HandleABCDxACBE2AECD(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 012:reshape+reshape+matmul+reshape-->reshape+batchmatmul
  Status HandleABCDxECD2ABE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 014: transpose+batchmatmul+transpose
  Status HandleABCDxACBE2ADBE(ge::ComputeGraph &graph, ge::NodePtr &node);

  // 005 & 006
  Status HandleBatchMatmul(bool adj_x2, ge::ComputeGraph &graph, ge::NodePtr &node);

  std::shared_ptr<ge::OpDesc> CreateTransposeOpDesc(bool unknown_shape, const ge::NodePtr &node,
                                                    const std::string &op_name);
  std::shared_ptr<ge::OpDesc> CreateReshapeOpDesc(bool unknown_shape, const ge::NodePtr &node, uint32_t seq);

  bool SetTransposePerm(bool unknown_shape, const std::vector<int32_t> &perm, ge::ComputeGraph &graph,
                        std::shared_ptr<ge::OpDesc> &transpose_desc, ge::NodePtr &transpose_node);

  ge::NodePtr CreateReshapeNode(const std::vector<int64_t> &dims, ge::ComputeGraph &graph,
                                std::shared_ptr<ge::OpDesc> &reshape_desc, int32_t axis = 0, int32_t end_axis = 0);

  Status LinkNode(ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> &anchors, ge::NodePtr &node);
  void UnlinkAll(ge::NodePtr &node);

  Status RelinkMatmulNode(ge::NodePtr &origin_node, ge::NodePtr &input0, ge::NodePtr &input1, ge::NodePtr &matmul_node,
                          bool swap_input);

  using ProcFunc = Status (EinsumPass::*)(ge::ComputeGraph &graph, ge::NodePtr &node);
  const string kFusedOpType = "Einsum";
  static std::unordered_map<std::string, ProcFunc> staticShapeProcs_;
  static std::unordered_map<std::string, ProcFunc> dynamicShapeProcs_;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EINSUM_FUSION_PASS_H_
