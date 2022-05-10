/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file dynamic_rnn_v2_grad_fusion_pass.h
 * \brief DynamicRNNV2Grad fusion pass(DynamicRNNV2Grad --> LSTMIInputGrad &
 * LSTMWeightGrad(Split&Concat&Matmul&Reduce))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_v2_GRAD_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_v2_GRAD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicRNNV2GradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  ge::GeTensorDesc SetTensorDescription(const vector<int64_t>& dims, const ge::Format& format,
                                        const ge::DataType& dtype) const;
  ge::GeTensorDesc SetTensorDescription(const vector<int64_t>& dims, const ge::Format& format,
                                        const ge::DataType& dtype, const vector<int64_t>& ori_dims,
                                        const ge::Format& ori_format) const;
  void MakUpRange(ge::GeTensorDesc& desc, const vector<int64_t>& dims) const;
  ge::NodePtr AddNewNode(ge::ComputeGraph& graph, const ge::OpDescPtr& op_desc, vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicDxhMatMulNode(const ge::NodePtr& transdata_node, const ge::NodePtr& w_concat_node,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicDwMatMulNode(const ge::NodePtr& xh_node, const ge::NodePtr& transdata_node,
                                  ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  ge::OpDescPtr GetDynamicLSTMGradCellDesc(ge::OpDescPtr& fused_desc, ge::GeTensorDesc& curT_desc) const;
  ge::OpDescPtr CreateScaleConstDesc(const std::string& name, int32_t value) const;
  template <class T>
  ge::NodePtr CreateConstNode(const std::string& name, ge::GeTensorDesc& tensor_desc, vector<T>& const_data,
                              ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicAddLSTMInputGradNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                          vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicDbReduceSumNode(const ge::NodePtr& transdata_node, ge::NodePtr& fused_node,
                                     ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicWConcatNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicXHConcatNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                  vector<ge::NodePtr>& new_nodes) const;
  vector<int64_t> CreateSplitTensorDesc(ge::OpDescPtr& split_desc) const;
  ge::NodePtr DynamicXHSplitNode(const ge::NodePtr& dxh_matmul_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicDwSplitNode(const ge::NodePtr& dw_matmul_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DynamicUnsqueezeNode(const std::string& name, const ge::GeTensorDesc& input_desc,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr AddWTransdataNode(const ge::NodePtr& w_concat_node, ge::ComputeGraph& graph,
                                vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr AddDwTrasposeNode(const ge::NodePtr& dw_node, ge::ComputeGraph& graph,
                                vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr AddDwTrasDataNode(const ge::NodePtr& transpose_node, ge::ComputeGraph& graph,
                                vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr DbTransDataNode(const ge::NodePtr& sum_node, ge::ComputeGraph& graph,
                              vector<ge::NodePtr>& new_nodes) const;
  ge::NodePtr CreatePadNode(std::string& name, vector<int64_t>& src_dims, vector<int64_t>& dsc_dims,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) const;
  void GetNodeInfo(ge::NodePtr& fused_node);

  int64_t t_size = -1;
  int64_t batch_size = -1;
  int64_t batch_nz_size = 0;
  int64_t input_size = 0;
  int64_t input_nz_size = 0;
  int64_t hidden_size = 0;
  int64_t hidden_nz_size = 0;
  bool is_input_size_aligned = false;
  bool is_hidden_size_aligned = false;

  ge::DataType state_type = ge::DT_FLOAT16;
  std::string grad_name;
  std::string FUSED_OP_TYPE = "DynamicRNNV2GradFusionPass";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_v2_GRAD_FUSION_PASS_H
