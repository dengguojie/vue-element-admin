/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicAUGRUGrad fusion pass(DynamicAUGRUGrad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#ifndef FE_DYNAMIC_AUGRU_GRAD_FUSION_PASS_H
#define FE_DYNAMIC_AUGRU_GRAD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicAUGRUGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  void GetNodeInfo(ge::NodePtr dynamicGRUGradNode);
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
                             ge::NodePtr dynamicAUGRUGradNode, int64_t curT);
  ge::NodePtr AddOneHiddenGradNode(const string& gateOrder, int64_t curT, ge::NodePtr dynamicAUGRUGradNode,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode, ge::NodePtr dynamicAUGRUGradNode,
                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  vector<vector<ge::NodePtr>> AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes, ge::NodePtr dynamicAUGRUGradNode,
                                           ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTConcatNode(const string& nodeName, const string& inputName, vector<int64_t> fzDims,
                             ge::NodePtr dynamicAUGRUGradNode, vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);

  // dw_h  matmul(h.T, dgate_h)
  map<std::string, ge::NodePtr> AddGRUHiddenGradNode(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddHTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddHSplitNode(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                            bool& failStatus);
  ge::NodePtr AddDwhTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                              bool& failStatus);
  ge::NodePtr AddHConcatNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr hSplitNode, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr hConcatNode, ge::NodePtr gruHiddenGradNode,
                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTConcatNodeBack(const string& nodeName, const string& inputName, vector<int64_t> fzDims,
                                 ge::NodePtr dynamicAUGRUGradNode, vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& newNodes, bool& failStatus);
  // for time_step = 1
  ge::NodePtr AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes, bool& failStatus);
  // concate dit, drt, dnt, to dgate_x
  ge::NodePtr AddDgateHSplitNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDgateXConcatNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateHSplitNode,
                                  ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                  bool& failStatus);
  // dx_t matmul(dgate_x, w_x.T)
  Status AddDxtMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr gateConcatNode, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& newNodes);
  // dw_x matmul(x.T, dgate_x)
  ge::NodePtr AddDwxMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr gateConcatNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode, int anchorIndex,
                               const vector<int64_t>& axis, const string& nodeName, const string& indexName,
                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr GetConstNodeOne(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                              const vector<int64_t>& axis, ge::ComputeGraph& graph,
                              vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddTMatMulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                             ge::NodePtr constNode, const string& nodeName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                             bool& failStatus);
  ge::NodePtr AddTReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                int anchorIndex,  const vector<int64_t>& axis, const string& nodeName,
                                ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  Status AddDwReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dwxMatmulNode, ge::NodePtr dwhMatmulNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  Status AddDbReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dbxNode, ge::NodePtr dbhNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  ge::NodePtr AddTConcatNodeDnxBack(const string& nodeName, const string& inputName,
                                    vector<int64_t> fzDims, ge::NodePtr dynamicAUGRUGradNode,
                                    vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                    vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDgateHSplitNodeBack(ge::NodePtr dynamicAUGRUGradNode,
                                     ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                     vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddDgateXConcatNodeBack(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateHSplitNode,
                                      ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                      vector<ge::NodePtr>& newNodes, bool& failStatus);
  Status AddDxtMatmulNodeBack(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                              ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes);
  ge::NodePtr AddDwxMatmulNodeBack(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                   bool& failStatus);
  Status AddDwAttReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dwAttNode, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& newNodes);
  const string FUSED_OP_TYPE = "AUGRUHiddenGradCell_Split_Concat_Matmul_Reduce";
  int64_t t_size = 0;
  int64_t batch = 0;
  int64_t nzBatch = 0;
  int64_t input_dim = 0;
  int64_t nzInputDim = 0;
  int64_t hidden_dim = 0;
  int64_t nzHiddenDim = 0;
  int64_t splitSize = 2;
  int64_t fzDim = 16;
  ge::DataType inputHType = ge::DT_FLOAT;
  bool fusion_reduce = false;
};
}  // namespace fe

#endif  // FE_DYNAMIC_AUGRU_GRAD_FUSION_PASS_H
