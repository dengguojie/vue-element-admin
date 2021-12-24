/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2Grad fusion pass(DynamicGRUV2Grad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#ifndef FE_DYNAMIC_GRU_V2_GRAD_D_FUSION_PASS_H
#define FE_DYNAMIC_GRU_V2_GRAD_D_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicGRUV2GradDFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  void GetNodeInfo(ge::NodePtr dynamicGRUGradNode);
  void AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                        const ge::Format& format, const vector<int64_t>& originDims, const ge::Format& originFormat,
                        const ge::DataType& dtype);
  void AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                        const ge::Format& format, const ge::DataType& dtype);
  void AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                        const ge::Format& format, const vector<int64_t>& originDims, const ge::Format& originFormat,
                        const ge::DataType& dtype, std::vector<std::pair<int64_t, int64_t>> x_range);
  void AddOutputNodeDesc(ge::OpDescPtr opDesc, const string& name, const vector<int64_t>& dims,
                         const ge::DataType& dtype, const ge::Format& format);
  void AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                         const ge::Format& format, const vector<int64_t>& originDims, const ge::Format& originFormat,
                         const ge::DataType& dtype);
  ge::NodePtr AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, vector<ge::NodePtr>& newNodes,
                         bool& failStatus);
  void AddT0GradNodeEdge(map<std::string, ge::NodePtr>& inputNodes, ge::NodePtr hiddenGradNode,
                         ge::NodePtr matmulGradNode, ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                         ge::NodePtr dynamicGRUGradNode);
  ge::NodePtr AddOneHiddenGradNode(const string& gateOrder, ge::NodePtr tSizeConst, ge::NodePtr dynamicGRUGradNode,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddT0MatmulNode(ge::NodePtr hiddenGradNode, ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                              vector<ge::NodePtr>& newNodes, bool& failStatus);
  vector<ge::NodePtr> AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes, ge::NodePtr dynamicGRUGradNode,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTConcatNode(const string& nodeName, const string& inputName, vector<int64_t> fzDims,
                             ge::NodePtr dynamicGRUGradNode, vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr BuildT0Cell(const string& gateOrder, ge::GeTensorDesc tStateDesc, ge::NodePtr dynamicGRUGradNode,
                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddInputReshapeNode(ge::NodePtr dynamicGRUGradNode, string reshapeName, ge::GeTensorDesc inputDesc,
                                  ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  // dw_h  matmul(h.T, dgate_h)
  map<std::string, ge::NodePtr> AddGRUHiddenGradNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr BuildSubNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& totalT, ge::ComputeGraph& graph,
                           bool& failStatus);
  ge::NodePtr BuildSizeConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& subNode, ge::ComputeGraph& graph,
                                  bool& failStatus);
  ge::NodePtr AddHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& sizeConcatNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes);
  ge::NodePtr AddHConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hSplitNode, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hConcatNode, ge::NodePtr gruHiddenGradNode,
                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  // concate dit, drt, dnt, to dgate_x
  ge::NodePtr AddDgateHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gruHiddenGradNode, ge::NodePtr whileNode,
                                 ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDgateXConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateHSplitNode,
                                  ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                  bool& failStatus);
  // dx_t matmul(dgate_x, w_x.T)
  Status AddDxtMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gateConcatNode, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& newNodes);
  // dw_x matmul(x.T, dgate_x)
  ge::NodePtr AddDwxMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gateConcatNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode, int anchorIndex,
                               const vector<int64_t>& axis, const string& nodeName, const string& indexName,
                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  Status AddDwReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dwxMatmulNode, ge::NodePtr dwhMatmulNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  Status AddDbReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dbxNode, ge::NodePtr dbhNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  vector<ge::NodePtr> BuildWhileNodes(ge::ComputeGraph& graph, ge::NodePtr dynamicGRUGradNode, ge::NodePtr t0Cell,
                                      ge::NodePtr t0Matmul, ge::NodePtr currTConst, ge::NodePtr tSizeConst,
                                      ge::GeTensorDesc concatHDesc, ge::GeTensorDesc concatXDesc);
  ge::ComputeGraphPtr BuildCondGraph(ge::NodePtr whileNode, int32_t argNum);
  ge::GeTensorDesc BuildTensorDesc(const vector<int64_t>& dims, const ge::Format& format,
                                   const vector<int64_t>& originDims, const ge::Format& originFormat,
                                   const ge::DataType& dtype);
  ge::OpDescPtr CreateConstDesc(const std::string& name, int32_t value, const std::string& dtype);
  ge::ComputeGraphPtr BuildBodyGraph(ge::NodePtr& whileNode, int32_t argNum,
                                     ge::NodePtr dynamicGRUGradNode);
  ge::OpDescPtr buildConcatDesc(const std::string& nodeName,
                                ge::NodePtr dynamicGRUGradNode, ge::GeTensorDesc input0Desc,
                                ge::GeTensorDesc input1Desc, ge::GeTensorDesc outputDesc);
  ge::ComputeGraphPtr BuildIfThenGraph(ge::NodePtr ifNode, ge::NodePtr dynamicGRUGradNode, ge::NodePtr& whileNode);
  ge::ComputeGraphPtr BuildElseBranchGraph(ge::NodePtr ifNode, ge::NodePtr dynamicGRUGradNode, ge::NodePtr& whileNode);
  ge::OpDescPtr CreateListConstDesc(const std::string& name, std::vector<int64_t> values);
  ge::OpDescPtr AddBodyMatmulNode(const string& nodeName, ge::GeTensorDesc inputDesc, ge::NodePtr dynamicGRUGradNode);
  ge::OpDescPtr buildCellDesc(const string& cellName, const string& gateOrder, ge::NodePtr& whileNode);
  ge::NodePtr BuildUnique(std::string name, vector<ge::NodePtr>& newNodes, bool& failStatus, ge::ComputeGraph& graph);
  ge::NodePtr BuildGather(std::string name, ge::GeTensorDesc inputTensorDescH, ge::GeTensorDesc indicesDesc,
                          ge::GeTensorDesc outDesc, vector<ge::NodePtr>& newNodes, bool& failStatus,
                          ge::ComputeGraph& graph);
  ge::NodePtr BuildShape(ge::GeTensorDesc xDesc, ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                         vector<ge::NodePtr>& newNodes, bool& failStatus);

  const string FUSED_OP_TYPE = "GRUV2HiddenGradCell_Split_Concat_Matmul_Reduce";
  int64_t t_size = 0;
  int64_t batch = 0;
  int64_t nzBatch = 0;
  int64_t batch_start = 1;
  int64_t batch_end = 32;
  int64_t input_dim = 0;
  int64_t nzInputDim = 0;
  int64_t hidden_dim = 0;
  int64_t nzHiddenDim = 0;
  ge::DataType inputHType = ge::DT_FLOAT;
  bool fusion_reduce = false;
};
}  // namespace fe

#endif  // FE_DYNAMIC_GRU_V2_GRAD_D_FUSION_PASS_H
