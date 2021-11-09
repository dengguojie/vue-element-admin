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
 * \file dynamic_rnn_grad_d_align_fusion_pass.h
 * \brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Split&Concat&Matmul&Reduce))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_D_ALIGN_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_D_ALIGN_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class DynamicRNNGradDAlignFusionPass : public PatternFusionBasePass {
  protected:
    vector<FusionPattern *> DefinePatterns() override;

    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;

  private:
    ge::OpDescPtr CreateConstDesc(const std::string &name, int32_t value, const std::string &dtype);

    ge::OpDescPtr CreateListConstDesc(const std::string &name, std::vector<int64_t> values);

    ge::ComputeGraphPtr BuildCondGraph(ge::NodePtr &whileNode, int32_t argNum);

    ge::OpDescPtr GetDynamicLSTMGradCellNode(std::string cellNodeName, ge::NodePtr dynamicRNNGradNode,
                                             ge::GeTensorDesc curTDesc,
                                             ge::ComputeGraph &graph, bool &failStatus);

    ge::OpDescPtr GetDynamicMatMulNode(std::string matmulNodeName, ge::NodePtr dynamicRNNGradNode,
                                       ge::ComputeGraph &graph, bool &failStatus, ge::GeShape dgateShape);

    vector<ge::OpDescPtr> GetDynamicSplitNode(std::string splitNodeName, std::string splitDimNodeName,
                                              std::string splitSizeNodeName,
                                              ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph, bool &failStatus,
                                              ge::GeTensorDesc matmulOutputDesc);

    ge::OpDescPtr GetDynamicBodyDxConcatNode(std::string cellNodeName, ge::NodePtr dynamicRNNGradNode,
                                             ge::ComputeGraph& graph, bool& failStatus,
                                             ge::GeTensorDesc splitInputDesc, ge::GeTensorDesc concatOriDesc);

    ge::OpDescPtr GetDynamicDxConcatNode(std::string cellNodeName, ge::NodePtr dynamicRNNGradNode,
                                         ge::ComputeGraph &graph, bool &failStatus,
                                         ge::GeTensorDesc splitInputDesc, ge::GeTensorDesc concatOriDesc);

    ge::ComputeGraphPtr BuildBodyGraph(ge::ComputeGraph &graph, ge::NodePtr &whileNode, int32_t argNum,
                                       ge::NodePtr dynamicRNNGradNode, ge::GeTensorDesc concatOriDesc,
                                       ge::GeTensorDesc concatDgateOriDesc, bool &failStatus);

    vector<ge::NodePtr> BuildWhileNodes(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph,
                                        vector<ge::NodePtr> &newNodes, bool &failStatus,
                                        ge::GeTensorDesc concatOriDesc,
                                        ge::GeTensorDesc concatDgateOriDesc,
                                        ge::GeTensorDesc curTDesc, ge::GeTensorDesc tDesc,
                                        ge::GeTensorDesc reshapeDxDesc,
                                        ge::GeTensorDesc reshapeDgateDesc);

    vector<ge::NodePtr> BuildT0Graph(ge::NodePtr dynamicRNNGradNode, ge::GeTensorDesc curTDesc, ge::ComputeGraph &graph,
                                     vector<ge::NodePtr> &newNodes, bool &failStatus);

    vector<ge::NodePtr> DynamicAddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph,
                                                    vector<ge::NodePtr> &newNodes, bool &failStatus);

    Status DynamicAddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &while_node,
                                     ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes);

    Status DynamicAddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                     ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes);

    ge::NodePtr DynamicAddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                     ge::NodePtr &while_node, ge::ComputeGraph &graph,
                                     vector<ge::NodePtr> &newNodes, bool &failStatus);

    ge::NodePtr DynamicAddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode,
                                     ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                     bool &failStatus);

    vector<ge::NodePtr> GetDynamicReshapeNode(std::string &reshapeNodeName, ge::NodePtr dynamicRNNGradNode,
                                              ge::NodePtr dgateInput,
                                              ge::GeTensorDesc outputDesc, ge::NodePtr shapeNode,
                                              ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr GetDynamicReshapeDxNode(std::string &reshapeNodeName, ge::NodePtr dynamicRNNGradNode,
                                        ge::GeTensorDesc inputDesc, ge::GeTensorDesc outputDesc,
                                        ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr DynamicAddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                      ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                      bool &failStatus);

    ge::NodePtr BuildSubNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &tSplitNode,
                             ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr BuildSizeConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &subNode,
                                    ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr
    DynamicAddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &sizeConcatNode, ge::ComputeGraph &graph,
                        vector<ge::NodePtr> &newNodes, bool &failStatus);

    ge::NodePtr BuildTShape(ge::GeTensorDesc xDesc, ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph,
                            bool &failStatus);

    ge::NodePtr BuildTSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph,
                            bool &failStatus);

    ge::NodePtr DynamicAddConcatHCNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &sizeConcatNode,
                                       ge::ComputeGraph &graph,
                                       vector<ge::NodePtr> &newNodes, bool &failStatus);

    ge::NodePtr DynamicAddInputReshapeNode(ge::NodePtr dynamicRNNGradNode, string reshapeName,
                                           ge::GeTensorDesc inputDesc, ge::ComputeGraph &graph,
                                           vector<ge::NodePtr> &newNodes, bool &failStatus);

    ge::NodePtr DynamicAddInithReshapeNode(ge::NodePtr dynamicRNNGradNode, string reshapeName,
                                           ge::GeTensorDesc inputDesc, ge::ComputeGraph &graph,
                                           vector<ge::NodePtr> &newNodes, bool &failStatus);

    vector<ge::OpDescPtr> GetDynamicBodyReshapeNode(std::string &reshapeNodeName, std::string &reshapeConstNodeName,
                                                    ge::NodePtr dynamicRNNGradNode, ge::GeTensorDesc inputDesc,
                                                    ge::GeTensorDesc outputDesc,
                                                    ge::ComputeGraph &graph, bool &failStatus);

    vector<ge::OpDescPtr> GetDynamicBodyDxReshapeNode(std::string &reshapeNodeName, std::string &reshapeConstNodeName,
                                                      ge::NodePtr dynamicRNNGradNode, ge::GeTensorDesc inputDesc,
                                                      ge::GeTensorDesc outputDesc,
                                                      ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr BuildDxReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode, std::string &nodeName,
                                             ge::NodePtr &negOneNode, ge::NodePtr &inputSizeNode,
                                             ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr BuildTDgateSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode,
                                 ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr BuildDgateReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode, std::string &nodeName,
                                                ge::NodePtr &subNode, ge::ComputeGraph &graph, bool &failStatus);

    ge::NodePtr AddTransposeNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                 bool &failStatus);

    ge::NodePtr AddDxPadNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                             bool &failStatus);

    ge::OpDescPtr
    AddTransposeToRNNNode(std::string transposeNodeName, ge::NodePtr dynamicRNNGradNode, bool &failStatus);

    const string FUSED_OP_TYPE = "DynamicRNNGradDAlignFusionPass";

    int64_t t_size = -1;
    int64_t batch_size = -1;
    int64_t input_size = 0;
    int64_t hidden_size = 0;

    int64_t batch_nz_size = -1;
    int64_t input_nz_size = 0;
    int64_t hidden_nz_size = 0;
    int64_t batch_start = 1;
    int64_t batch_end = 32;

    ge::NodePtr reshapeInitC = nullptr;
    ge::NodePtr reshapeInitH = nullptr;
    ge::NodePtr reshapeDh = nullptr;
    ge::NodePtr reshapeDc = nullptr;
    string DynamicRNNGradName;
  };

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_D_ALIGN_FUSION_PASS_H_
