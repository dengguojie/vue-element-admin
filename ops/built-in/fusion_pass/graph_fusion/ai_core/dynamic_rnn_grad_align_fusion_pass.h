/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file diag_part_fusion_pass.h
 * \brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Split&Concat&Matmul&Reduce))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicRNNGradAlignFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;

  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status AddEdgeForCell(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                        bool& failStatus, vector<vector<ge::NodePtr>> resultNode, ge::NodePtr lstmSplitC,
                        ge::NodePtr lstmSplitDy, ge::NodePtr lstmSplitI, ge::NodePtr lstmSplitJ, ge::NodePtr lstmSplitF,
                        ge::NodePtr lstmSplitO, ge::NodePtr lstmSplitTanh, ge::NodePtr lstmXConcatD,
                        ge::NodePtr& lstmGageConcatD);

  vector<vector<ge::NodePtr>> AddTLoopNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                           vector<ge::NodePtr>& newNodes, bool& failStatus);

  ge::NodePtr AddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                   vector<ge::NodePtr>& newNodes, bool& failStatus);

  ge::NodePtr AddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                           bool& failStatus);

  ge::NodePtr AddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);

  ge::NodePtr AddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, bool& failStatus);

  ge::NodePtr AddConcatNodeT_1(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                               bool& failStatus);

  ge::NodePtr AddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode, ge::NodePtr lstmInputGradNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);

  Status AddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes);

  Status AddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr lstmInputGradNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, ge::NodePtr const_one_node);

  ge::NodePtr GetConstNodeOne(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                              bool& failStatus);

  ge::NodePtr AddTransposeNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                           bool& failStatus);

  const string FUSED_OP_TYPE = "LSTMInputGrad_Split_Concat_Matmul_Reduce";
  int64_t input_dim = 0;
  int64_t input_nz_dim = 0;
  int64_t hidden_dim = 0;
  int64_t hidden_nz_dim = 0;
  int64_t batch_dim = 0;
  int64_t batch_nz_dim = 0;
  int64_t t_dim = 0;
  int64_t tSizeJudge = 0;
  int64_t cIdx0 = 0;
  int64_t cIdx1 = 1;

  void AddBatchMatMulForCell(const ge::GeShape &output_origin_shape, ge::OpDescPtr &lstmBatchMatMulDesc,
                             vector<int64_t> &outputy_dims) const;

  ge::OpDescPtr &AddSpiltForCell(vector<int64_t> &outputy_dims, ge::OpDescPtr &lstmSplitDesc) const;

  void SetInputDescForGradCell(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &inputC,
                               const ge::GeTensorDesc &inputDy, const ge::GeTensorDesc &inputI, int64_t i,
                               ge::OpDescPtr &basicLstmCellStateGradDesc) const;

  void SetOutputDescForGradCell(const ge::GeTensorDesc &inputI, ge::OpDescPtr &basicLstmCellStateGradDesc,
                                vector<int64_t> &output_dims, ge::GeShape &output_origin_shape,
                                ge::GeTensorDesc &output_tensor_desc) const;

  void SetReshapeDescForCell(const vector<int64_t> &output_dims, const ge::GeTensorDesc &output_tensor_desc,
                             ge::OpDescPtr &reshape_desc) const;

  ge::OpDescPtr &SetDescForSplitVDI(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                    int64_t num_split_x, ge::OpDescPtr &lstmSplitIDesc) const;

  ge::OpDescPtr &SetDescForSplitVDJ(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                    int64_t num_split_x, ge::OpDescPtr &lstmSplitJDesc) const;

  ge::OpDescPtr &SetDescForSplitVDF(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                    int64_t num_split_x, ge::OpDescPtr &lstmSplitFDesc) const;

  ge::OpDescPtr &SetDescForSplitVDO(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                    int64_t num_split_x, ge::OpDescPtr &lstmSplitODesc) const;

  ge::OpDescPtr &
  SetDescForSplitVDTanh(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                        int64_t num_split_x, ge::OpDescPtr &lstmSplitTanhDesc) const;

  ge::OpDescPtr &SetDescForSplitVDC(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                    int64_t num_split_x, ge::OpDescPtr &lstmSplitCDesc) const;

  ge::OpDescPtr &SetDescForSplitVDdy(const ge::OpDescPtr &dynamicRNNGradDesc, const ge::GeTensorDesc &split_tensor_desc,
                                     int64_t num_split_x, ge::OpDescPtr &lstmSplitDyDesc) const;

  ge::OpDescPtr &
  SetDescForDgateConcatD(const vector<vector<ge::NodePtr>> &result_node, const ge::OpDescPtr &dynamicRNNGradDesc,
                         int64_t num_split_x, ge::OpDescPtr &lstmGageConcatDDesc) const;

  ge::OpDescPtr &
  SetDescForxConcatD(const vector<vector<ge::NodePtr>> &result_node, const ge::OpDescPtr &dynamicRNNGradDesc,
                     int64_t num_split_x, ge::OpDescPtr &lstmXConcatDDesc) const;

  void AddEdgeForSplitNode(const ge::NodePtr &dynamicRNNGradNode, const ge::NodePtr &lstmSplitC,
                           const ge::NodePtr &lstmSplitDy, const ge::NodePtr &lstmSplitI, const ge::NodePtr &lstmSplitJ,
                           const ge::NodePtr &lstmSplitF, const ge::NodePtr &lstmSplitO,
                           const ge::NodePtr &lstmSplitTanh) const;

  ge::GeTensorDesc CreateTensorDescForSplit(const ge::OpDescPtr &dynamicRNNGradDesc) const;

  ge::GeTensorDesc SetOutputDescForGradCell(const ge::GeTensorDesc &inputI, ge::OpDescPtr &basicLstmCellStateGradDesc,
                                            const ge::GeShape &output_shape) const;

  vector<int64_t> getOutputDimsForGradCell(const ge::OpDescPtr &basicLstmCellStateGradDesc) const;

  ge::OpDescPtr &SetDescForTransdataDb(ge::OpDescPtr &transdataDbDesc) const;

  ge::OpDescPtr &SetDescForTransdataDw(ge::OpDescPtr &transdataDwDesc) const;
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_
