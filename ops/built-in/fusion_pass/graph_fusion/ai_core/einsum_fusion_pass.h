/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
  Status CheckProduct(const std::vector<int64_t> &shape) const;
  Status CheckInputArgs(const Mapping &mapping, bool &is_dynamic_shape) const;

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

  std::shared_ptr<ge::OpDesc> CreateTransposeOpDesc(const ge::NodePtr &node,
                                                    const std::string &op_name);
  std::shared_ptr<ge::OpDesc> CreateReshapeOpDesc(bool unknown_shape, const ge::NodePtr &node, uint32_t seq);

  bool SetTransposePerm(const std::vector<int32_t> &perm, ge::ComputeGraph &graph,
                        std::shared_ptr<ge::OpDesc> &transpose_desc, ge::NodePtr &transpose_node);

  ge::NodePtr CreateReshapeNode(const std::vector<int64_t> &dims, ge::ComputeGraph &graph,
                                std::shared_ptr<ge::OpDesc> &reshape_desc, int32_t axis = 0, int32_t end_axis = 0);

  Status LinkEinsumOutputNode(const ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> &anchors,
                              const ge::NodePtr &node) const;
  void UnlinkAllDataAnchors(const ge::NodePtr &node) const;

  Status RelinkMatmulNode(ge::NodePtr &origin_node, ge::NodePtr &input0, ge::NodePtr &input1, ge::NodePtr &matmul_node,
                          bool swap_input);

  enum EinsumDimensionType {
    BROAD_CAST = 0,  // ...
    BATCH,           // exist in both two inputs, also in output
    FREE,            // exist in both two inputs, not in output
    CONTRACT,        // only exist in one inputs, also in output
    REDUCE,          // only exist in one inputs, not in output
    DIM_TYPE_NUM
  };

  struct LabelInfo {
    std::string labels;
    std::vector<int32_t> indices;
  };

  using LabelCount = std::map<char, uint32_t>;
  using DimensionType2LabelInfo = std::vector<LabelInfo>;
  void ResetFusionPass();
  Status SplitOpInFuzzScene(const std::string &equation, ge::ComputeGraph &graph, ge::NodePtr &node);
  Status ParseEquation(const std::string &equation, std::vector<std::string> &in_equations, std::string &out_equation);

  void SplitStr2Vector(const std::string &input, const std::string &delimiter, std::vector<std::string> &output) const;

  void CountLabels(const std::string &equation, LabelCount &label_count, std::set<char> &labels) const;

  void CollectDimensionType(size_t dim_num, const std::string &equation, DimensionType2LabelInfo &label_info) const;

  void MapDimensionType(const std::set<char> &labels, const LabelCount &input0_label_count,
                        const LabelCount &input1_label_count, const LabelCount &output_label_count);

  void ReorderAxes(DimensionType2LabelInfo &label_infos) const;

  void CompareAxes(EinsumDimensionType dim_type, const DimensionType2LabelInfo &output_label_info,
                   DimensionType2LabelInfo &input_label_info) const;

  bool GetTransposeDstEquation(const std::string &ori_equation, const DimensionType2LabelInfo &label_infos,
                               std::vector<int32_t> &perm_list, bool &input_free_contract_order,
                               std::string &dst_equation) const;

  void CheckMergeFreeLabels(const ge::NodePtr &node, const std::string &out_equation);

  void CheckBatchMatmulSwapInputs();

  void CalcBatchMatmulOutput(size_t input_num, const ge::NodePtr &node, bool &need_reshape,
                             std::string &bmm_out_equation, std::vector<int64_t> &bmm_dims) const;

  ge::GeTensorDescPtr GetPrevOutputDescAfterBmm(const ge::NodePtr &node) const;

  ge::GeTensorDescPtr GetPrevOutputDesc(const ge::NodePtr &node, size_t idx) const;

  Status TransposeInput(const std::vector<std::string> &in_equations, const ge::NodePtr &node, ge::ComputeGraph &graph);

  Status TransposeOutput(const std::string &out_equation, const ge::NodePtr &node,
                         ge::ComputeGraph &graph, std::string &cur_out_equation);

  Status StrideInput(const std::vector<std::string> &in_equations) const;

  Status InflatedOutput(const std::string &out_equation) const;

  Status ReduceInput(const std::vector<std::string> &in_equations, const ge::NodePtr &node, ge::ComputeGraph &graph);

  Status ReshapeInput(const std::vector<std::string> &in_equations, const std::string &out_equation,
                      const ge::NodePtr &node, ge::ComputeGraph &graph);

  Status ReshapeOutput(const std::string &out_equation, const ge::NodePtr &node,
                       ge::ComputeGraph &graph, std::string &cur_out_equation,
                       const std::vector<std::string> &in_equations);

  Status DoBatchMatmul(const std::vector<std::string> &in_equations, const ge::NodePtr &node, ge::ComputeGraph &graph);

  Status LinkEinsumInputNode(const ge::NodePtr &node, const ge::NodePtr &first_node, int32_t anchor_idx,
                             int32_t first_node_anchor_idx) const;
  Status LinkContinuousNodes(const std::vector<ge::NodePtr> &nodes) const;

  Status ReLinkCtrlEdges(const ge::NodePtr &node, const ge::NodePtr &last_node) const;

  Status ReLinkNodes(const ge::NodePtr &node);

  const std::vector<std::string> kDimensionType2Str = {"BroadCast", "Batch", "Free", "Contract", "Reduce"};

  using ProcFunc = Status (EinsumPass::*)(ge::ComputeGraph &graph, ge::NodePtr &node);
  static std::unordered_map<std::string, ProcFunc> dynamicShapeProcs_;

  uint32_t transpose_seq = 1;
  uint32_t reduce_seq = 1;
  uint32_t reshape_seq = 1;
  uint32_t batchmatmul_seq = 1;
  bool swap_bmm_inputs = false;
  bool merge_free_labels = true;
  std::map<char, EinsumDimensionType> dim_types_map;
  std::vector<bool> input_free_contract_orders;
  std::vector<std::vector<ge::NodePtr>> bmm_input_nodes;
  ge::NodePtr batchmatmul_node;
  std::vector<ge::NodePtr> bmm_output_nodes;

  std::vector<DimensionType2LabelInfo> input_label_infos;
  DimensionType2LabelInfo output_label_info;
  DimensionType2LabelInfo ori_output_label_info;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EINSUM_FUSION_PASS_H_
