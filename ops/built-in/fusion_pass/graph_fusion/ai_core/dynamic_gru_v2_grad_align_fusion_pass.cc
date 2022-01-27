/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2Grad fusion pass(DynamicGRUV2Grad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_gru_v2_grad_align_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "DynamicGRUV2Grad";
static const std::string PATTERN_FUSEDNODE = "DynamicGRUV2Grad";
static map<std::string, int> INPUT_INDEX = {
    {"x", 0},           {"weight_input", 1}, {"weight_hidden", 2}, {"y", 3},     {"init_h", 4}, {"h", 5},
    {"dy", 6},          {"dh", 7},           {"update", 8},        {"reset", 9}, {"new", 10},   {"hidden_new", 11},
    {"seq_length", 12}, {"mask", 13}};

static map<std::string, int> HIDDENGRAD_INPUT_INDEX = {{"dh_pre_t", 0}, {"h", 1},     {"dy", 2},  {"dh", 3},
                                                       {"update", 4},   {"reset", 5}, {"new", 6}, {"hidden_new", 7}};
static map<std::string, int> OUTPUT_INDEX = {{"dw_input", 0},  {"dw_hidden", 1}, {"db_input", 2},
                                             {"db_hidden", 3}, {"dx", 4},        {"dh_prev", 5}};
static map<std::string, int> HIDDENGRAD_OUTPUT_INDEX = {{"dh_prev", 0}, {"dgate_h", 1}, {"dnt_x", 2}};
static int64_t splitSize = 2;
static int64_t fzDim = 16;
vector<FusionPattern*> DynamicGRUV2GradAlignFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicGRUV2GradAFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                       "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicGRUV2GradAlignFusionPass::GetNodeInfo(ge::NodePtr dynamicGRUGradNode) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  t_size = inputTensorDescH.GetShape().GetDim(0);
  batch = inputTensorDescH.GetShape().GetDim(1);
  nzBatch = (batch + fzDim - 1) / fzDim;
  hidden_dim = inputTensorDescH.GetShape().GetDim(2);
  nzHiddenDim = (hidden_dim + 15) / 16;

  ge::GeTensorDesc inputTensorDescX = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  input_dim = inputTensorDescX.GetShape().GetDim(2);
  nzInputDim = (input_dim + fzDim - 1) / fzDim;
  inputHType = inputTensorDescH.GetDataType();
  return;
}

void DynamicGRUV2GradAlignFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                       const vector<int64_t>& dims, const ge::Format& format,
                                                       const vector<int64_t>& originDims,
                                                       const ge::Format& originFormat, const ge::DataType& dtype) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  opDesc->AddInputDesc(name, addNodeDesc);
  return;
}

void DynamicGRUV2GradAlignFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                        const vector<int64_t>& dims, const ge::DataType& dtype,
                                                        const ge::Format& format) {
  ge::GeShape originShape(dims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(format);
  opDesc->AddOutputDesc(name, addNodeDesc);
  return;
}

void DynamicGRUV2GradAlignFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                        const vector<int64_t>& dims, const ge::Format& format,
                                                        const vector<int64_t>& originDims, const ge::Format& originFormat,
                                                        const ge::DataType& dtype) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginDataType(dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  opDesc->AddOutputDesc(name, addNodeDesc);
  return;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc,
                                                        vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::NodePtr node = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(node);
  return node;
}

void DynamicGRUV2GradAlignFusionPass::AddHiddenGradNodeEdge(map<std::string, ge::NodePtr>& inputNodes,
                                                            ge::NodePtr hiddenGradNode, ge::NodePtr matmulGradNode,
                                                            ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                                                            ge::NodePtr dynamicGRUGradNode, int64_t curT) {
  if (curT == 0) {
    // fake connect dh_pre_t
    ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh_pre_t"]));
    // connect dh
    ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh"]));
  } else {
    // connect dh_pre(last)->dh_pre_t
    ge::GraphUtils::AddEdge(lastHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dh_prev"]),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh_pre_t"]));
    // connect matmul(last)->dh
    ge::GraphUtils::AddEdge(lastMatmulNode->GetOutDataAnchor(0),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh"]));
  }
  // connect h
  if (curT < t_size - 1) {
    ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  } else {
    // fake connect th last cell
    ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  }

  // connect dh_prev to output
  if (curT == t_size) {
    for (InDataAnchorPtr inAnchorPtr :
         dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dh_prev"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dh_prev"]), inAnchorPtr);
    }
  }

  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dy"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["update"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["update"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["reset"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["reset"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["new"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["new"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["hidden_new"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["hidden_new"]));
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddOneHiddenGradNode(const string& gateOrder, int64_t curT,
                                                                  ge::NodePtr dynamicGRUGradNode,
                                                                  ge::ComputeGraph& graph,
                                                                  vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::OpDescPtr hiddenGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (hiddenGradDesc = std::make_shared<ge::OpDesc>(
           dynamicGRUGradNode->GetName() + "/GRUV2Grad/GRUV2HiddenGradCell_" + std::to_string(curT),
           "GRUV2HiddenGradCell")),
      hiddenGradDesc = nullptr;
      failStatus = true;
      return nullptr);

  // set attr of gate order
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);
  // set attr of t_state
  ge::AttrUtils::SetInt(hiddenGradDesc, "t_state", curT);

  // set input desc
  ge::GeTensorDesc dhPrevDesc = dynamicGRUGradDesc->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).Clone();
  hiddenGradDesc->AddInputDesc("dh_pre_t", dhPrevDesc);
  if (curT < t_size - 1) {
    hiddenGradDesc->AddInputDesc("h", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]).Clone());
  } else {
    hiddenGradDesc->AddInputDesc("init_h", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["init_h"]).Clone());
  }
  hiddenGradDesc->AddInputDesc("dy", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dy"]).Clone());
  ge::GeTensorDesc dhDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dh"]).Clone();
  if (curT == 0) {
    hiddenGradDesc->AddInputDesc("dh", dhDesc);
  } else {
    AddInputNodeDesc(hiddenGradDesc, "dh", {1, nzHiddenDim, nzBatch, 16, 16}, ge::FORMAT_FRACTAL_NZ,
                     {1, batch, hidden_dim}, ge::FORMAT_ND, inputHType);
  }
  hiddenGradDesc->AddInputDesc("update", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["update"]).Clone());
  hiddenGradDesc->AddInputDesc("reset", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["reset"]).Clone());
  hiddenGradDesc->AddInputDesc("new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["new"]).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());

  vector<int64_t> dgateHNzDim{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dgateHNzDimOri{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDim{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDimOri{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  ge::Format dgateOriFormat = ge::FORMAT_FRACTAL_NZ;
  ge::Format dnxOriFormat = ge::FORMAT_FRACTAL_NZ;

  hiddenGradDesc->AddOutputDesc("dh_prev", dhPrevDesc);
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHNzDim, ge::FORMAT_FRACTAL_NZ, dgateHNzDimOri, dgateOriFormat,
                    inputHType);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateNzDim, ge::FORMAT_FRACTAL_NZ, singleGateNzDimOri,
                    dnxOriFormat, inputHType);

  // create gru_hidden_grad node
  ge::NodePtr hiddenGradNode = this->AddNewNode(graph, hiddenGradDesc, newNodes, failStatus);
  return hiddenGradNode;
}

void DynamicGRUV2GradAlignFusionPass::AddBatchMatMulForCell(ge::OpDescPtr& lstmBatchMatMulDesc,
                                                            const string &weightName) {  // add matmul input
  vector<int64_t> LeftDims{nzHiddenDim * (splitSize + 1), nzBatch, fzDim, fzDim};
  vector<int64_t> LeftOriDims{batch, nzHiddenDim * (splitSize + 1) * fzDim};
  vector<int64_t> WeightDims{nzHiddenDim, nzHiddenDim * (splitSize + 1), fzDim, fzDim};
  vector<int64_t> WeightoriDims{hidden_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputDims{nzHiddenDim, nzBatch, fzDim, fzDim};

  if (weightName == "weight_input") {
    LeftDims = {t_size, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    LeftOriDims = {t_size, batch, nzHiddenDim * (splitSize + 1) * fzDim};
    WeightDims = {nzInputDim, nzHiddenDim * (splitSize + 1), fzDim, fzDim};
    WeightoriDims = {input_dim, 3 * hidden_dim};
    outputDims = {t_size, nzInputDim, nzBatch, fzDim, fzDim};
  }

  GeTensorDesc left_tensor_desc = GeTensorDesc(GeShape(LeftDims), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  left_tensor_desc.SetOriginShape(GeShape(LeftOriDims));
  left_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddInputDesc("dgate", left_tensor_desc);

  GeTensorDesc weight_tensor_desc = GeTensorDesc(GeShape(WeightDims), FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16);
  weight_tensor_desc.SetOriginShape(GeShape(WeightoriDims));
  weight_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddInputDesc("w", weight_tensor_desc);

  // add matmul output
  GeShape outputOriShape(outputDims);
  GeShape outputShape(outputDims);
  GeTensorDesc outputTensorDesc = GeTensorDesc(outputShape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  outputTensorDesc.SetOriginShape(outputOriShape);
  outputTensorDesc.SetOriginFormat(FORMAT_FRACTAL_NZ);
  lstmBatchMatMulDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x1", false);
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x2", true);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "input_size", input_dim);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "hidden_size", hidden_dim);
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode,
                                                                        ge::NodePtr dynamicGRUGradNode,
                                                                        ge::ComputeGraph& graph,
                                                                        vector<ge::NodePtr>& newNodes,
                                                                        bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (matmulDesc = std::make_shared<ge::OpDesc>(
         dynamicGRUGradNode->GetName() + "/GRUV2Grad/Matmul_" + to_string(curT), "BatchMatMulV2")),
    matmulDesc = nullptr;
    failStatus = true;
    return nullptr);

  AddBatchMatMulForCell(matmulDesc, "weight_hidden");

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                          matmulNode->GetInDataAnchor(0));  // dgate_h
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));  // weight
  return matmulNode;
}

vector<vector<ge::NodePtr>> DynamicGRUV2GradAlignFusionPass::AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes,
                                                                          ge::NodePtr dynamicGRUGradNode,
                                                                          ge::ComputeGraph& graph,
                                                                          vector<ge::NodePtr>& newNodes,
                                                                          bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();

  string gateOrder = "zrh";
  ge::AttrUtils::GetStr(dynamicGRUGradDesc, "gate_order", gateOrder);

  vector<vector<ge::NodePtr>> result = {};
  vector<ge::NodePtr> hiddenGradNodes = {};
  vector<ge::NodePtr> matmulNodes = {};
  ge::NodePtr lastHiddenGradNode = nullptr;
  ge::NodePtr lastMatmulNode = nullptr;

  for (int64_t i = 0; i < t_size; i++) {
    ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, i, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    ge::NodePtr matmulNode =
        AddOneHiddenGradMatmulNode(i, hiddenGradNode, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    // add input edge
    AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, matmulNode, lastHiddenGradNode, lastMatmulNode,
                          dynamicGRUGradNode, i);

    lastHiddenGradNode = hiddenGradNode;
    lastMatmulNode = matmulNode;
    hiddenGradNodes.push_back(hiddenGradNode);
    matmulNodes.push_back(matmulNode);
  }
  // last hiddenGradNode
  ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, t_size, dynamicGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return result);
  AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, nullptr, lastHiddenGradNode, lastMatmulNode, dynamicGRUGradNode,
                        t_size);
  hiddenGradNodes.push_back(hiddenGradNode);

  result.push_back(hiddenGradNodes);
  result.push_back(matmulNodes);
  return result;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddTConcatNode(const string& nodeName, const string& inputName,
                                                            vector<int64_t> fzDims, ge::NodePtr dynamicGRUGradNode,
                                                            vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + nodeName, "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  FUSION_PASS_CHECK(srcNodes.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                     "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);

  GeTensorDesc inputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  for (int64_t i = 0; i < t_size; i++) {
    concatDesc->AddInputDesc("input_" + to_string(i), inputDesc);
  }

  // output concat, shape:{t,batch_size,hidden_size}
  GeTensorDesc outputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> outDim = outputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(outDim.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);
  outDim[0] = t_size;
  outputDesc.SetShape(GeShape(outDim));
  outputDesc.SetOriginShape(GeShape(outDim));
  concatDesc->AddOutputDesc("concat_" + inputName, outputDesc);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", t_size);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return nullptr);

  // Edge
  for (int64_t i = 0; i < t_size; i++) {
    ge::GraphUtils::AddEdge(srcNodes[i]->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX[inputName]),
                            concatNode->GetInDataAnchor(t_size - 1 - i));  // Init_h
  }
  return concatNode;
}

map<std::string, ge::NodePtr> DynamicGRUV2GradAlignFusionPass::AddGRUHiddenGradNode(ge::NodePtr dynamicGRUGradNode,
                                                                                    ge::ComputeGraph& graph,
                                                                                    vector<ge::NodePtr>& newNodes,
                                                                                    bool& failStatus) {
  map<std::string, ge::NodePtr> inputNodes;
  map<std::string, ge::NodePtr> result;
  vector<vector<ge::NodePtr>> result_node;
  if (t_size > 1) {
    // add loop t hidden grad nodes; [ [hidden_grad_nodes] [matmul_nodes] ]
    result_node = AddTLoopNode(inputNodes, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    FUSION_PASS_CHECK(result_node.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                          "result_node is null, fusion failed."),
                      return result);
    FUSION_PASS_CHECK(result_node[0].empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                             "result_node is null, fusion failed."),
                      return result);

    // add dnt_x concat node by t
    vector<int64_t> fzDims = {1, nzHiddenDim, nzBatch, 16, 16};
    ge::NodePtr dntXConcatTNode = nullptr;
    dntXConcatTNode = AddTConcatNode("/GRUV2Grad/ConcatDntX", "dnt_x", fzDims, dynamicGRUGradNode,
                                     result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "AddDntXConcatTNode:check failed, fusion failed."),
                      return result);

    // add dgate_h concat node
    fzDims = {1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    ge::NodePtr dgateHConcatTNode = nullptr;
    dgateHConcatTNode = AddTConcatNode("/GRUV2Grad/ConcatDgateH", "dgate_h", fzDims, dynamicGRUGradNode,
                                       result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "AddDgateHConcatTNode:check failed, fusion failed."),
                      return result);

    result["dgate_h"] = dgateHConcatTNode;
    result["dnt_x"] = dntXConcatTNode;
  } else {
    result_node = AddTLoopNode(inputNodes, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."),
                      return result);
    FUSION_PASS_CHECK(result_node.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                          "result_node is null, fusion failed."),
                      return result);
    FUSION_PASS_CHECK(result_node[0].empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                             "result_node is null, fusion failed."),
                      return result);
    ge::NodePtr node = result_node[0][0];
    result["dgate_h"] = node;
    result["dnt_x"] = node;
  }
  ge::NodePtr dhPrevNode = result_node[0][result_node[0].size() - 1];
  result["dh_prev"] = dhPrevNode;
  return result;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddHTransData(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr transDataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transDataDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUweightGrad/Dwh/HTransData",
                                                    "TransData")),
      transDataDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone();
  transDataDesc->AddInputDesc("trans_src", inputTensorDescH);

  // output
  vector<int64_t> dstDims = {t_size, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(transDataDesc, "trans_dst", dstDims, inputTensorDescH.GetDataType(), ge::FORMAT_FRACTAL_NZ);

  // attr
  ge::AttrUtils::SetStr(transDataDesc, "src_format", "ND");
  ge::AttrUtils::SetStr(transDataDesc, "dst_format", "FRACTAL_NZ");

  // create node
  ge::NodePtr transNode = AddNewNode(graph, transDataDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          transNode->GetInDataAnchor(0));
  return transNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/SplitVD",
                                                "SplitVD")),
      splitDesc = nullptr;
      failStatus = true;
      return nullptr);

  // add transData
  ge::NodePtr transNode = AddHTransData(dynamicGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);
  // add input
  ge::GeTensorDesc inputTensorDescH = transNode->GetOpDesc()->GetOutputDesc(0).Clone();
  splitDesc->AddInputDesc("input_h", inputTensorDescH);

  vector<int64_t> size_splits = {(t_size - 1) * batch, batch};
  // add output1 split_t_1, shape:{t-1,batch_size,hidden_size}
  vector<int64_t> output1NzDim{t_size - 1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_t_1", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_t_1

  // add output2 split_1, shape:{1,batch_size,hidden_size}
  vector<int64_t> output2NzDim{1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_1", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_1
  // attr
  size_splits = {t_size - 1, 1};

  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", splitSize);

  // create split node
  ge::NodePtr splitNode = AddNewNode(graph, splitDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), splitNode->GetInDataAnchor(0));

  return splitNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDwhTransData(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr transDataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transDataDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUweightGrad/Dwh/TransData",
                                                    "TransData")),
      transDataDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  inputTensorDescInitH.SetShape(GeShape({1, batch, hidden_dim}));
  transDataDesc->AddInputDesc("trans_src", inputTensorDescInitH);
  // output
  vector<int64_t> dstDims = {1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(transDataDesc, "trans_dst", dstDims, inputTensorDescInitH.GetDataType(), ge::FORMAT_FRACTAL_NZ);

  // attr
  ge::AttrUtils::SetStr(transDataDesc, "src_format", "ND");
  ge::AttrUtils::SetStr(transDataDesc, "dst_format", "FRACTAL_NZ");

  // create node
  ge::NodePtr transNode = AddNewNode(graph, transDataDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          transNode->GetInDataAnchor(0));

  return transNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddHConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr splitNode,
                                                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                            bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/HConcatD",
                                                 "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::NodePtr transNode = AddDwhTransData(dynamicGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);
  ge::GeTensorDesc inputTensorDescInitH = transNode->GetOpDesc()->GetOutputDesc(0).Clone();

  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output concat_h, shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDim{t_size, batch, hidden_dim};
  vector<int64_t> outputNzDim{t_size, nzHiddenDim, nzBatch, 16, 16};

  AddOutputNodeDesc(concatDesc, "concat_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  return concatNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hConcatNode,
                                                              ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
  (matmulDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                             "BatchMatMul")),
    matmulDesc = nullptr;
    failStatus = true;
    return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  // gruHiddenGradNode is dgateConcatNode
  ge::GeTensorDesc inputTensorDescDgate;
  if (t_size == 1) {
    inputTensorDescDgate =
        gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();  // dgate_h
  } else {
    inputTensorDescDgate = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dgate_h
  }
  inputTensorDescH.SetDataType(ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(GeShape({t_size, batch, (splitSize + 1) * hidden_dim}));

  inputTensorDescDgate.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{t_size, hidden_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};

  AddOutputNodeDesc(matmulDesc, "dwt_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));  // ht
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            matmulNode->GetInDataAnchor(1));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_h
  }
  return matmulNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode,
                                                              ge::NodePtr gruHiddenGradNode,
                                                              ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                              bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                                 "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  // gruHiddenGradNode is dgateConcatNode
  ge::GeTensorDesc inputTensorDescDgate;
  if (t_size == 1) {
    inputTensorDescDgate =
        gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();  // dgate_h
  } else {
    inputTensorDescDgate = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }

  inputTensorDescH.SetShape(GeShape({1, nzHiddenDim, nzBatch, 16, 16}));
  inputTensorDescH.SetOriginShape(GeShape({1, batch, hidden_dim}));
  inputTensorDescH.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescH.SetDataType(ge::DT_FLOAT16);

  inputTensorDescDgate.SetShape(GeShape({1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim}));
  inputTensorDescDgate.SetOriginShape(GeShape({1, batch, (splitSize + 1) * hidden_dim}));
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{t_size, nzHiddenDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  if (t_size == 1) {
    outputDim = {nzHiddenDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
    outputNzDim = {(splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  }

  AddOutputNodeDesc(matmulDesc, "dwt_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));  // Init_h
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            matmulNode->GetInDataAnchor(1));  // dgate_h
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_h
  }
  return matmulNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDgateHSplitNode(ge::NodePtr dynamicGRUGradNode,
                                                                ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                                vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/SplitVD",
                                                "SplitVD")),
      splitDesc = nullptr;
      failStatus = true;
      return nullptr);

  // add input
  ge::GeTensorDesc inputDgateH;
  if (t_size == 1) {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  } else {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }
  splitDesc->AddInputDesc("input_dgate_h", inputDgateH);

  // add output1 dgate_ir, shape:{t, batch, 2 * hidden_size}
  vector<int64_t> output1Dim{t_size, batch, splitSize * hidden_dim};
  vector<int64_t> output1NzDim{t_size, splitSize * nzHiddenDim, nzBatch, fzDim, fzDim};
  AddOutputNodeDesc(splitDesc, "split_ir", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_didr

  // add output2 split_1, shape:{t, batch, hidden_size}
  vector<int64_t> output2NzDim{t_size, nzHiddenDim, nzBatch, fzDim, fzDim};
  AddOutputNodeDesc(splitDesc, "split_n", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_dn_h

  // attr
  vector<int64_t> size_splits = {splitSize * nzHiddenDim, nzHiddenDim};
  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 1);
  ge::AttrUtils::SetInt(splitDesc, "num_split", splitSize);

  // create split node
  ge::NodePtr splitNode = AddNewNode(graph, splitDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            splitNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), splitNode->GetInDataAnchor(0));
  }
  return splitNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDgateXConcatNode(ge::NodePtr dynamicGRUGradNode,
                                                                 ge::NodePtr dgateHSplitNode,
                                                                 ge::NodePtr gruHiddenGradNode,
                                                                 ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes,
                                                                 bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD",
                                                 "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  vector<int64_t> dirtNzDesc = {t_size, splitSize * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dnxNzDesc = {t_size, nzHiddenDim, nzBatch, fzDim, fzDim};
  AddInputNodeDesc(concatDesc, "input_dirt", dirtNzDesc, ge::FORMAT_FRACTAL_NZ, dirtNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);
  AddInputNodeDesc(concatDesc, "input_dnt_x", dnxNzDesc, ge::FORMAT_FRACTAL_NZ, dnxNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);

  // output shape:{t,batch,3*hidden_size}
  vector<int64_t> outputDim{t_size, batch, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  AddOutputNodeDesc(concatDesc, "dgate_x", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 1);
  ge::AttrUtils::SetInt(concatDesc, "N", splitSize);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dgateHSplitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));  // [dit, drt]
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dnt_x"]),
                            concatNode->GetInDataAnchor(1));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  }
  return concatNode;
}

Status DynamicGRUV2GradAlignFusionPass::AddDxtMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulOpDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmulV2",
                                                   "BatchMatMulV2")),
      matmulOpDesc = nullptr;
      return false);

  AddBatchMatMulForCell(matmulOpDesc, "weight_input");
  // create matmul node
  bool failStatus = false;
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulOpDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));  // dgate_x
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_input"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));

  // output Edge
  for (InDataAnchorPtr inAnchorPtr : dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
    // dxt
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return failStatus;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDwxMatmulNode(ge::NodePtr dynamicGRUGradNode,
                                                              ge::NodePtr dgateXConcatNode,
                                                              ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes,
                                                              bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/BatchMatmul",
                                                 "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc xtDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).Clone();  // xt
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();              // dgate_x
  xtDesc.SetDataType(ge::DT_FLOAT16);
  xtDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  xtDesc.SetOriginFormat(ge::FORMAT_ND);
  xtDesc.SetShape(GeShape({t_size, nzInputDim, nzBatch, fzDim, fzDim}));
  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  dgateXDesc.SetOriginFormat(ge::FORMAT_ND);
  dgateXDesc.SetOriginShape(GeShape({t_size, batch, (splitSize + 1) * nzHiddenDim * fzDim}));
  matmulDesc->AddInputDesc("xt", xtDesc);
  matmulDesc->AddInputDesc("dgate_x", dgateXDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim{t_size, nzInputDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim};
  if (t_size == 1) {
    outputDim = {nzInputDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
    outputNzDim = {(splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim};
  }
  AddOutputNodeDesc(matmulDesc, "dwt_x", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // input Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));                                         // xt
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x
  return matmulNode;
}

ge::OpDescPtr DynamicGRUV2GradAlignFusionPass::SetDescForTransdata(ge::OpDescPtr &transdataDesc,
                                                                   const string& srcFormat,
                                                                   const string& weightName) {
  // input for transdata
  int64_t dim0 = input_dim;
  int64_t nzdim0 = nzInputDim;
  if (weightName == "weight_hidden") {
    dim0 = hidden_dim;
    nzdim0 = nzHiddenDim;
  }
  vector<int64_t> transDims{nzdim0, nzHiddenDim * 3, 16, 16};
  vector<int64_t> transOriDims{dim0, hidden_dim * 3};
  ge::Format transFormat = ge::FORMAT_FRACTAL_ZN_RNN;
  if (srcFormat == "ND_RNN_BIAS") {
    transDims = {nzHiddenDim * 3 * 16};
    transOriDims = {hidden_dim * 3};
    transFormat = ge::FORMAT_ND_RNN_BIAS;
  }
  GeTensorDesc transInputDesc =
      GeTensorDesc(GeShape(transDims), transFormat, DT_FLOAT16);
  transInputDesc.SetOriginShape(GeShape(transOriDims));
  transInputDesc.SetOriginFormat(FORMAT_ND);
  transdataDesc->AddInputDesc("trans_src", transInputDesc);
  // output for tarnsdata
  GeTensorDesc transOutputDesc =
      GeTensorDesc(GeShape(transOriDims), FORMAT_ND, DT_FLOAT16);
  transOutputDesc.SetOriginShape(GeShape(transOriDims));
  transOutputDesc.SetOriginFormat(FORMAT_ND);
  transdataDesc->AddOutputDesc("trans_dsc", transOutputDesc);
  // attr
  AttrUtils::SetStr(transdataDesc, "src_format", srcFormat);
  AttrUtils::SetStr(transdataDesc, "dst_format", "ND");
  AttrUtils::SetInt(transdataDesc, "input_size", input_dim);
  AttrUtils::SetInt(transdataDesc, "hidden_size", hidden_dim);
  return transdataDesc;
}

ge::OpDescPtr DynamicGRUV2GradAlignFusionPass::SetDescForTranspose(ge::OpDescPtr &transposeDesc,
                                                                   const string &weightName) {
  // input for transdata
  int64_t dim0 = input_dim;
  int64_t nzdim0 = nzInputDim;
  if (weightName == "weight_hidden") {
    dim0 = hidden_dim;
    nzdim0 = nzHiddenDim;
  }
  vector<int64_t> transInputDims{nzHiddenDim * 3, nzdim0, 16, 16};
  vector<int64_t> transInputOriDims{nzdim0 * 16, nzHiddenDim * 16 * 3};
  vector<int64_t> transOutputDims{nzdim0, nzHiddenDim * 3, 16, 16};
  vector<int64_t> transOutputOriDims{dim0, hidden_dim * 3};

  // input for transpose
  GeTensorDesc transInputDesc =
      GeTensorDesc(GeShape(transInputDims), ge::FORMAT_FRACTAL_NZ, DT_FLOAT16);
  transInputDesc.SetOriginShape(GeShape(transInputOriDims));
  transInputDesc.SetOriginFormat(FORMAT_ND);
  transposeDesc->AddInputDesc("x", transInputDesc);

  // output for transpose
  GeTensorDesc transOutputDesc =
      GeTensorDesc(GeShape(transOutputDims), ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16);
  transOutputDesc.SetOriginShape(GeShape(transOutputOriDims));
  transOutputDesc.SetOriginFormat(FORMAT_ND);
  transposeDesc->AddOutputDesc("y", transOutputDesc);

  // attr
  vector<int32_t> permValue = {1, 0, 3, 2};
  ge::AttrUtils::SetListInt(transposeDesc, "perm", permValue);
  ge::AttrUtils::SetInt(transposeDesc, "input_size", input_dim);
  ge::AttrUtils::SetInt(transposeDesc, "hidden_size", hidden_dim);

  return transposeDesc;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDbReduceSumTransNode(
    ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode, int anchorIndex, const vector<int64_t>& axis,
    const string& nodeName, const string& indexName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
    const vector<int64_t>& transDims, bool& failStatus) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(
           dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD")),
      reduceSumDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  reduceSumDesc->AddInputDesc("input_" + nodeName, inputTensorDesc);

  ge::Format transFormat = ge::FORMAT_ND_RNN_BIAS;
  // output
  ge::GeTensorDesc reduceOutputTensorDesc = ge::GeTensorDesc(GeShape(transDims), transFormat, ge::DT_FLOAT16);
  vector<int64_t> outputOriDims{hidden_dim * 3};
  reduceOutputTensorDesc.SetOriginShape(GeShape(outputOriDims));
  reduceOutputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("output_" + nodeName, reduceOutputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", axis);
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // trans_data_rnn 
  ge::OpDescPtr transdataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdataDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransDataRNN", "TransDataRNN")),
      return nullptr);
  transdataDesc = SetDescForTransdata(transdataDesc, "ND_RNN_BIAS", "bias");
  ge::NodePtr transNode = this->AddNewNode(graph, transdataDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);
  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), reduceSumNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), transNode->GetInDataAnchor(0));

  for (InDataAnchorPtr inAnchorPtr :
       dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddDwReduceSumTransNode(
    ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode, int anchorIndex, const vector<int64_t>& axis,
    const string& nodeName, const string& indexName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
    const vector<int64_t>& transDims, const string& weightName, bool& failStatus) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(
           dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD")),
      reduceSumDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  reduceSumDesc->AddInputDesc("input_" + nodeName, inputTensorDesc);

  // output
  ge::GeTensorDesc reduceOutputTensorDesc = ge::GeTensorDesc(GeShape(transDims), ge::FORMAT_FRACTAL_NZ,
                                                             ge::DT_FLOAT16);
  int64_t weightDim0 = nzInputDim;
  if (weightName == "weight_hidden") {
    weightDim0 = nzHiddenDim;
  }
  vector<int64_t> outputOriDims{weightDim0 * 16, nzHiddenDim * 16 * 3};
  reduceOutputTensorDesc.SetOriginShape(GeShape(outputOriDims));
  reduceOutputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("output_" + nodeName, reduceOutputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", axis);
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // transpose
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransPose", "TransposeD")),
      return nullptr);
  transposeDesc = SetDescForTranspose(transposeDesc, weightName);
  ge::NodePtr transposeNode = this->AddNewNode(graph, transposeDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), reduceSumNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), transposeNode->GetInDataAnchor(0));

  for (InDataAnchorPtr inAnchorPtr :
       dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddTransposeNode(
    ge::NodePtr dynamicGRUGradNode, ge::NodePtr dwMatmulNode, const string& nodeName, const string& weightName,
    const string& outputName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // insert transpose
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransPose", "TransposeD")),
      return nullptr);
  transposeDesc = SetDescForTranspose(transposeDesc, weightName);
  bool failStatus = false;
  ge::NodePtr transposeNode = this->AddNewNode(graph, transposeDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "check failed, fusion failed."),
                    return nullptr);
  ge::GraphUtils::AddEdge(dwMatmulNode->GetOutDataAnchor(0), transposeNode->GetInDataAnchor(0));
  for (InDataAnchorPtr inAnchorPtr :
      dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[outputName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return transposeNode;
}

ge::NodePtr DynamicGRUV2GradAlignFusionPass::AddTReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode,
                                                               int anchorIndex,  const vector<int64_t>& axis,
                                                               const string& nodeName,
                                                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                               bool& failStatus) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(
           dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD")),
      reduceSumDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  reduceSumDesc->AddInputDesc("input_" + nodeName, inputTensorDesc);

  // gen output dims
  vector<int64_t> outputDim = inputTensorDesc.GetShape().GetDims();
  for (int64_t i: axis) {
    outputDim[i] = 1;
  }

  // output
  AddOutputNodeDesc(reduceSumDesc, "output_" + nodeName, outputDim, inputHType, inputTensorDesc.GetFormat());

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", axis);
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), reduceSumNode->GetInDataAnchor(0));
  return reduceSumNode;
}

Status DynamicGRUV2GradAlignFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dwxMatmulNode,
                                                           ge::NodePtr dwhMatmulNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes) {
  // add dw_x / dw_h reduce_sum
  if (t_size == 1) {
    AddTransposeNode(dynamicGRUGradNode, dwxMatmulNode, "dwx", "weight_input", "dw_input", graph, newNodes);
    AddTransposeNode(dynamicGRUGradNode, dwhMatmulNode, "dwh", "weight_hidden", "dw_hidden", graph, newNodes);

    return SUCCESS;
  }
  int anchorOutputIndex = 0;
  vector<int64_t> reduceDwAxis{0};
  bool isFailure = false;

  AddDwReduceSumTransNode(dynamicGRUGradNode, dwxMatmulNode, anchorOutputIndex, reduceDwAxis, "dwx", "dw_input",
                          graph, newNodes, {3 * nzHiddenDim, nzInputDim, fzDim, fzDim}, "weight_input", isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxReduceSumNode:check failed, fusion failed."),
                    return FAILED);
  
  AddDwReduceSumTransNode(dynamicGRUGradNode, dwhMatmulNode, anchorOutputIndex, reduceDwAxis, "dwh", "dw_hidden",
                          graph, newNodes, {3 * nzHiddenDim, nzHiddenDim, fzDim, fzDim}, "weight_hidden", isFailure);  
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

Status DynamicGRUV2GradAlignFusionPass::AddDbReduceSumNode(ge::NodePtr gruV2GradNode, ge::NodePtr dbxNode,
                                                           ge::NodePtr dbhNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes) {
  // add db_x / db_h reduce_sum
  int anchorOutputIndex = 0;
  bool isFailure = false;
  vector<int64_t> reduceDbAxis{2, 3};
  if (t_size == 1) {
    // NZ {1, 3 * nzHiddenDim, nzBatch, 16, 16}
    // ND {1, batch, 3*hidden}
    vector<int64_t> reduceDbxAxis{1};
    AddDbReduceSumTransNode(gruV2GradNode, dbxNode, anchorOutputIndex, reduceDbAxis, "dbx", "db_input",
                            graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbxReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    anchorOutputIndex = 1;
    AddDbReduceSumTransNode(gruV2GradNode, dbhNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden",
                            graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbhReduceSumNode:check failed, fusion failed."),
                      return FAILED);
    return SUCCESS;
  }
  // {t_size, 3 * nzHiddenDim, nzBatch, 16, 16}
  vector<int64_t> reduceDbTAxis{0};
  // dbx
  ge::NodePtr dbxTReduceSumNode = AddTReduceSumNode(gruV2GradNode, dbxNode, anchorOutputIndex, reduceDbTAxis,
                                                    "dbx_t", graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbxTReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddDbReduceSumTransNode(gruV2GradNode, dbxTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbx", "db_input",
                          graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  // dbh
  ge::NodePtr dbhTReduceSumNode = AddTReduceSumNode(gruV2GradNode, dbhNode, anchorOutputIndex, reduceDbTAxis,
                                                    "dbh_t", graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbhTReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddDbReduceSumTransNode(gruV2GradNode, dbhTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden",
                          graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

Status DynamicGRUV2GradAlignFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradAlignFusionPass fusion begin.");
  bool isFailure = false;
  // get dynamicGRUGradNode
  ge::NodePtr gruV2GradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(gruV2GradNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicGRUV2Grad:grad node is null, fusion failed."),
                    return FAILED);
  ge::OpDescPtr dynamicGRUGradDesc = gruV2GradNode->GetOpDesc();
  FUSION_PASS_CHECK(dynamicGRUGradDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicGRUV2Grad:op desc is null, fusion failed."),
                    return FAILED);
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  ge::GeTensorDesc inputTensorDescX = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  batch = inputTensorDescH.GetShape().GetDim(1);
  hidden_dim = inputTensorDescH.GetShape().GetDim(splitSize);
  input_dim = inputTensorDescX.GetShape().GetDim(splitSize);

  t_size = inputTensorDescH.GetShape().GetDim(0);
  if (batch % fzDim == 0 && hidden_dim % fzDim == 0 && input_dim % fzDim == 0 && t_size != 1) {
    fusion_reduce = true;
  }
  if (hidden_dim % 16 == 0 && input_dim % 16 == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is 16 align, will not changed");
    return NOT_CHANGED;
  }
  fusion_reduce = false;
  if (PatternFusionUtil::IsUnknownShape(batch) ||
      PatternFusionUtil::IsUnknownShape(hidden_dim) || PatternFusionUtil::IsUnknownShape(t_size) ||
      PatternFusionUtil::IsUnknownShape(input_dim)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "DynamicGRUV2GradAlignFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (hidden_dim % fzDim == 0 && input_dim % fzDim == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is 16 align, will not changed.");
    return NOT_CHANGED;
  }

  // init shape
  this->GetNodeInfo(gruV2GradNode);

  // add gruHiddenGrad {dhPrevNode, dgateHConcatTNode, dntXConcatTNode}
  map<std::string, ge::NodePtr> hiddenGradNodes = AddGRUHiddenGradNode(gruV2GradNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddGRUHiddenGradNode:check failed, fusion failed."),
                    return FAILED);

  ge::NodePtr dwhMatmulNode;
  if (t_size != 1) {
    // add split
    ge::NodePtr splitNode = AddHSplitNode(gruV2GradNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddHSplitNode:check failed, fusion failed."),
                      return FAILED);

    // add concat
    ge::NodePtr hConcatNode = AddHConcatNode(gruV2GradNode, splitNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddHConcatNode:check failed, fusion failed."),
                      return FAILED);

    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode =
        AddDwhMatmulNode(gruV2GradNode, hConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  } else {
    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode = AddDwhMatmulNode(gruV2GradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  }

  // split dgate_h to [dit, drt] and [dnt_h]
  ge::NodePtr dgateHSplitNode = nullptr;
  dgateHSplitNode = AddDgateHSplitNode(gruV2GradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // concat [dit, drt] with [dnt_x] to dgate_x
  ge::NodePtr gateConcatNode = nullptr;
  gateConcatNode = AddDgateXConcatNode(gruV2GradNode, dgateHSplitNode,
                                       hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);

  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateXConcatNode:check failed, fusion failed."),
                    return FAILED);

  // add dxt matmul(dgate_x, w_x.T)
  isFailure = AddDxtMatmulNode(gruV2GradNode, gateConcatNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDxtMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x matmul(x.T, dgate_x)
  ge::NodePtr dwxMatmulNode = nullptr;
  dwxMatmulNode = AddDwxMatmulNode(gruV2GradNode, gateConcatNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxMatmulNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddDwReduceSumNode(gruV2GradNode, dwxMatmulNode, dwhMatmulNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddDbReduceSumNode(gruV2GradNode, gateConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  // unlink all control input of gruV2GradNode
  if (gruV2GradNode->GetInControlAnchor() != nullptr) {
    gruV2GradNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of gruV2GradNode
  for (auto inAnchor : gruV2GradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove gruV2GradNode from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(gruV2GradNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                     gruV2GradNode->GetName().c_str()),
      return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradAlignFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("DynamicGRUV2GradAlignFusionPass", BUILT_IN_GRAPH_PASS, DynamicGRUV2GradAlignFusionPass);
}  // namespace fe
