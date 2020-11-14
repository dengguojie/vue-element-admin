/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2Grad fusion pass(DynamicGRUV2Grad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#ifndef FE_DYNAMIC_GRU_V2_GRAD_FUSION_PASS_H
#define FE_DYNAMIC_GRU_V2_GRAD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicGRUV2GradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  void GetNodeInfo(ge::NodePtr node);
  void AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                        const ge::Format& format, const vector<int64_t>& originDims, const ge::Format& originFormat,
                        const ge::DataType& dtype);
  void AddOutputNodeDesc(ge::OpDescPtr opDesc, const string& name, const vector<int64_t>& dims,
                         const ge::DataType& dtype, const ge::Format& format);
  void AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name, const vector<int64_t>& dims,
                         const ge::Format& format, const vector<int64_t>& originDims, const ge::Format& originFormat,
                         const ge::DataType& dtype);
  ge::NodePtr AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc, vector<ge::NodePtr>& newNodes,
                         bool& failStatus);
  void AddHiddenGradNodeEdge(map<std::string, ge::NodePtr>& inputNodes, ge::NodePtr hiddenGradNode,
                             ge::NodePtr matmulGradNode, ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                             ge::NodePtr dynamicGRUGradNode, int64_t curT);
  void AddSplitTInputNodeDesc(ge::OpDescPtr hiddenGradDesc, ge::OpDescPtr dynamicGRUGradDesc, const std::string& name);
  int64_t GetTState(int64_t curT);
  ge::NodePtr AddOneHiddenGradNode(const string& gateOrder, int64_t curT, ge::NodePtr dynamicGRUGradNode,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode, ge::NodePtr dynamicGRUGradNode,
                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  vector<vector<ge::NodePtr>> AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes, ge::NodePtr dynamicGRUGradNode,
                                           ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTSplitNode(const string& nodeName, const string& inputName, ge::NodePtr dynamicGRUGradNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTConcatNode(const string& nodeName, const string& inputName, vector<int64_t> fzDims,
                             ge::NodePtr dynamicGRUGradNode, vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);

  // dw_h  matmul(h.T, dgate_h)
  map<std::string, ge::NodePtr> AddGRUHiddenGradNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                            bool& failStatus);
  ge::NodePtr AddDwhTransData(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                              bool& failStatus);
  ge::NodePtr AddHConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hSplitNode, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hConcatNode, ge::NodePtr gruHiddenGradNode,
                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  // for time_step = 1
  ge::NodePtr AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes, bool& failStatus);
  // concate dit, drt, dnt, to dgate_x
  ge::NodePtr AddDgateHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDgateXConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateHSplitNode,
                                  ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                  bool& failStatus);
  // dx_t matmul(dgate_x, w_x.T)
  ge::NodePtr AddWxBroadcastNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                 bool& failStatus);
  Status AddDxtMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gateConcatNode, ge::NodePtr wxBroadcastNode,
                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  // dw_x matmul(x.T, dgate_x)
  ge::NodePtr AddDwxMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gateConcatNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes, bool& failStatus);
  Status AddReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode, int anchorIndex,
                          const vector<int64_t>& axis, const string& nodeName, const string& indexName,
                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  ge::NodePtr AddNzTransDataNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode, int anchorIndex,
                                 const string& nodeName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                 bool& failStatus);
  const string FUSED_OP_TYPE = "GRUV2HiddenGradCell_Split_Concat_Matmul_Reduce";
  int64_t t_size = 0;
  int64_t batch = 0;
  int64_t nzBatch = 0;
  int64_t input_dim = 0;
  int64_t nzInputDim = 0;
  int64_t hidden_dim = 0;
  int64_t nzHiddenDim = 0;
  ge::DataType inputHType = ge::DT_FLOAT;
};
}  // namespace fe

#endif  // FE_DYNAMIC_GRU_V2_GRAD_FUSION_PASS_H
