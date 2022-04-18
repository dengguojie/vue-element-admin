/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicAUGRUGrad fusion pass(DynamicAUGRUGrad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_augru_grad_fusion_pass.h"

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
static const char* FUSED_NODE = "DynamicAUGRUGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicAUGRUGrad";
static map<std::string, int> INPUT_INDEX = {{"x",             0},
                                            {"weight_input",  1},
                                            {"weight_hidden", 2},
                                            {"weight_att",    3},
                                            {"y",             4},
                                            {"init_h",        5},
                                            {"h",             6},
                                            {"dy",            7},
                                            {"dh",            8},
                                            {"update",        9},
                                            {"update_att",    10},
                                            {"reset",         11},
                                            {"new",           12},
                                            {"hidden_new",    13},
                                            {"seq_length",    14},
                                            {"mask",          15}};

static map<std::string, int> HIDDENGRAD_INPUT_INDEX = {{"weight_att", 0},
                                                       {"dh_pre_t",   1},
                                                       {"h",          2},
                                                       {"dy",         3},
                                                       {"dh",         4},
                                                       {"update",     5},
                                                       {"update_att", 6},
                                                       {"reset",      7},
                                                       {"new",        8},
                                                       {"hidden_new", 9}};
static map<std::string, int> OUTPUT_INDEX = {{"dw_input",  0},
                                             {"dw_hidden", 1},
                                             {"db_input",  2},
                                             {"db_hidden", 3},
                                             {"dx",        4},
                                             {"dh_prev",   5},
                                             {"dw_att",    6}};
static map<std::string, int> HIDDENGRAD_OUTPUT_INDEX = {{"dh_prev",  0},
                                                        {"dgate_h",  1},
                                                        {"dnt_x",    2},
                                                        {"dw_att_t", 3}};

vector<FusionPattern*> DynamicAUGRUGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicAUGRUGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                       "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicAUGRUGradFusionPass::GetNodeInfo(ge::NodePtr dynamicAUGRUGradNode) {
  ge::OpDescPtr dynamicAUGRUGradDesc = dynamicAUGRUGradNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDescH = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  t_size = inputTensorDescH.GetShape().GetDim(0);
  batch = inputTensorDescH.GetShape().GetDim(1);
  nzBatch = (batch + fzDim - 1) / fzDim;
  hidden_dim = inputTensorDescH.GetShape().GetDim(2);
  nzHiddenDim = (hidden_dim + 15) / 16;

  ge::GeTensorDesc inputTensorDescX = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  input_dim = inputTensorDescX.GetShape().GetDim(2);
  nzInputDim = (input_dim + fzDim - 1) / fzDim;
  inputHType = inputTensorDescH.GetDataType();
  return;
}

void DynamicAUGRUGradFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                  const vector<int64_t>& dims, const ge::Format& format,
                                                  const vector<int64_t>& originDims, const ge::Format& originFormat,
                                                  const ge::DataType& dtype) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  opDesc->AddInputDesc(name, addNodeDesc);
  return;
}

void DynamicAUGRUGradFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

void DynamicAUGRUGradFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

ge::NodePtr DynamicAUGRUGradFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc,
                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::NodePtr node = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(node);
  return node;
}

void DynamicAUGRUGradFusionPass::AddHiddenGradNodeEdge(map<std::string, ge::NodePtr>& inputNodes,
                                                       ge::NodePtr hiddenGradNode, ge::NodePtr matmulGradNode,
                                                       ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                                                       ge::NodePtr dynamicAUGRUGradNode, int64_t curT) {
  if (curT == 0) {
    // fake connect dh_pre_t
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh_pre_t"]));
    // connect dh
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
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
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  } else {
    // fake connect th last cell
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  }

  // connect dh_prev to output
  if (curT == t_size) {
    for (InDataAnchorPtr inAnchorPtr :
         dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dh_prev"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dh_prev"]), inAnchorPtr);
    }
  }

  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_att"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["weight_att"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dy"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["update"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["update"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["update_att"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["update_att"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["reset"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["reset"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["new"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["new"]));
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["hidden_new"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["hidden_new"]));
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddOneHiddenGradNode(const string& gateOrder, int64_t curT,
                                                             ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicAUGRUGradDesc = dynamicAUGRUGradNode->GetOpDesc();
  ge::OpDescPtr hiddenGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (hiddenGradDesc = std::make_shared<ge::OpDesc>(
           dynamicAUGRUGradNode->GetName() + "/AUGRUGrad/AUGRUHiddenGradCell_" + std::to_string(curT),
           "AUGRUHiddenGradCell")),
      hiddenGradDesc = nullptr;
      failStatus = true;
      return nullptr);

  // set attr of gate order
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);
  // set attr of t_state
  ge::AttrUtils::SetInt(hiddenGradDesc, "t_state", curT);

  // set input desc
  hiddenGradDesc->AddInputDesc("weight_att", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["weight_att"]).Clone());
  ge::GeTensorDesc dhPrevDesc = dynamicAUGRUGradDesc->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).Clone();
  hiddenGradDesc->AddInputDesc("dh_pre_t", dhPrevDesc);
  if (curT < t_size - 1) {
    hiddenGradDesc->AddInputDesc("h", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]).Clone());
  } else {
    hiddenGradDesc->AddInputDesc("init_h", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["init_h"]).Clone());
  }
  hiddenGradDesc->AddInputDesc("dy", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["dy"]).Clone());
  ge::GeTensorDesc dhDesc = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["dh"]).Clone();
  if (curT == 0) {
    hiddenGradDesc->AddInputDesc("dh", dhDesc);
  } else {
    AddInputNodeDesc(hiddenGradDesc, "dh", {1, nzHiddenDim, nzBatch, 16, 16}, ge::FORMAT_FRACTAL_NZ,
                     {1, batch, hidden_dim}, ge::FORMAT_ND, inputHType);
  }
  hiddenGradDesc->AddInputDesc("update", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["update"]).Clone());
  hiddenGradDesc->AddInputDesc("update_att", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["update_att"]).Clone());
  hiddenGradDesc->AddInputDesc("reset", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["reset"]).Clone());
  hiddenGradDesc->AddInputDesc("new", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["new"]).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());

  vector<int64_t> dgateHNzDim{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dgateHNzDimOri{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDim{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDimOri{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dwAttTNzDim{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dwAttTNzDimOri{1, batch, nzHiddenDim * fzDim};
  ge::Format dgateOriFormat = ge::FORMAT_FRACTAL_NZ;
  ge::Format dnxOriFormat = ge::FORMAT_FRACTAL_NZ;
  if (fusion_reduce) {
    dgateHNzDim = {(splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    dgateHNzDimOri = {nzBatch * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
    singleGateNzDim = {nzHiddenDim, nzBatch, fzDim, fzDim};
    singleGateNzDimOri = {nzBatch * fzDim, nzHiddenDim * fzDim};
    dgateOriFormat = ge::FORMAT_ND;
    dnxOriFormat = ge::FORMAT_ND;
  }

  hiddenGradDesc->AddOutputDesc("dh_prev", dhPrevDesc);
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHNzDim, ge::FORMAT_FRACTAL_NZ, dgateHNzDimOri, dgateOriFormat,
                    inputHType);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateNzDim, ge::FORMAT_FRACTAL_NZ, singleGateNzDimOri,
                    dnxOriFormat, inputHType);
  AddOutputNodeDesc(hiddenGradDesc, "dw_att_t", dwAttTNzDim, ge::FORMAT_FRACTAL_NZ, dwAttTNzDimOri, ge::FORMAT_ND,
                    inputHType);

  // create gru_hidden_grad node
  ge::NodePtr hiddenGradNode = this->AddNewNode(graph, hiddenGradDesc, newNodes, failStatus);
  return hiddenGradNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode,
                                                                   ge::NodePtr dynamicAUGRUGradNode,
                                                                   ge::ComputeGraph& graph,
                                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  if (fusion_reduce) {
    FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(
           dynamicAUGRUGradNode->GetName() + "/AUGRUGrad/Matmul_" + to_string(curT), "MatMulV2")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);
  } else {
    FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(
           dynamicAUGRUGradNode->GetName() + "/AUGRUGrad/Matmul_" + to_string(curT), "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);
  }

  // input
  ge::GeTensorDesc inputDesc = hiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  inputDesc.SetDataType(ge::DT_FLOAT16);
  vector<int64_t> inputDim{1, batch, (splitSize + 1) * hidden_dim};
  vector<int64_t> inputNzDim{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  if (fusion_reduce) {
    inputDim = {batch, (splitSize + 1) * hidden_dim};
    inputNzDim = {(splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  }
  inputDesc.SetOriginShape(GeShape(inputDim));
  inputDesc.SetShape(GeShape(inputNzDim));
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);

  // weight
  ge::GeTensorDesc weightDesc = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone();
  weightDesc.SetOriginFormat(ge::FORMAT_ND);
  weightDesc.SetOriginShape(GeShape({1, hidden_dim, (splitSize + 1) * hidden_dim}));
  weightDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  vector<int64_t> weightNzDim = {1, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  if (fusion_reduce) {
    weightDesc.SetOriginShape(GeShape({hidden_dim, (splitSize + 1) * hidden_dim}));
    weightNzDim = {(splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  }
  weightDesc.SetShape(GeShape(weightNzDim));
  weightDesc.SetDataType(ge::DT_FLOAT16);

  matmulDesc->AddInputDesc("input_dgate_h", inputDesc);
  matmulDesc->AddInputDesc("input_weight", weightDesc);

  vector<int64_t> outputDim{1, batch, hidden_dim};
  vector<int64_t> outputNzDim{1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(matmulDesc, "dh", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  if (fusion_reduce) {
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", false);
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", true);
  } else {
    ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
    ge::AttrUtils::SetBool(matmulDesc, "adj_x2", true);
  }

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                          matmulNode->GetInDataAnchor(0));  // dgate_h
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));  // weight
  return matmulNode;
}

vector<vector<ge::NodePtr>> DynamicAUGRUGradFusionPass::AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes,
                                                                     ge::NodePtr dynamicAUGRUGradNode,
                                                                     ge::ComputeGraph& graph,
                                                                     vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicAUGRUGradDesc = dynamicAUGRUGradNode->GetOpDesc();

  string gateOrder = "zrh";
  ge::AttrUtils::GetStr(dynamicAUGRUGradDesc, "gate_order", gateOrder);

  vector<vector<ge::NodePtr>> result = {};
  vector<ge::NodePtr> hiddenGradNodes = {};
  vector<ge::NodePtr> matmulNodes = {};
  ge::NodePtr lastHiddenGradNode = nullptr;
  ge::NodePtr lastMatmulNode = nullptr;

  for (int64_t i = 0; i < t_size; i++) {
    ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, i, dynamicAUGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    ge::NodePtr matmulNode =
        AddOneHiddenGradMatmulNode(i, hiddenGradNode, dynamicAUGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    // add input edge
    AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, matmulNode, lastHiddenGradNode, lastMatmulNode,
                          dynamicAUGRUGradNode, i);

    lastHiddenGradNode = hiddenGradNode;
    lastMatmulNode = matmulNode;
    hiddenGradNodes.push_back(hiddenGradNode);
    matmulNodes.push_back(matmulNode);
  }
  // last hiddenGradNode
  ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, t_size, dynamicAUGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return result);
  AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, nullptr, lastHiddenGradNode, lastMatmulNode, dynamicAUGRUGradNode,
                        t_size);
  hiddenGradNodes.push_back(hiddenGradNode);

  result.push_back(hiddenGradNodes);
  result.push_back(matmulNodes);
  return result;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddTConcatNodeDnxBack(const string& nodeName, const string& inputName,
                                                              vector<int64_t> fzDims, ge::NodePtr dynamicAUGRUGradNode,
                                                              vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + nodeName, "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  FUSION_PASS_CHECK(srcNodes.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                     "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);

  GeTensorDesc inputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> inputDims = {inputDesc.GetShape().GetDim(0), inputDesc.GetShape().GetDim(1),
                               inputDesc.GetShape().GetDim(2), inputDesc.GetShape().GetDim(3)};
  vector<int64_t> inputDimsOri = {inputDesc.GetShape().GetDim(1) * fzDim, inputDesc.GetShape().GetDim(0) * fzDim};
  inputDesc.SetShape(GeShape(inputDims));
  inputDesc.SetOriginShape(GeShape(inputDimsOri));
  for (int64_t i = 0; i < t_size; i++) {
    concatDesc->AddInputDesc("input_" + to_string(i), inputDesc);
  }

  // output concat, shape:{t,batch_size,hidden_size}
  GeTensorDesc outputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> outDim = {inputDims[0], t_size * inputDims[1], inputDims[2], inputDims[3]};
  vector<int64_t> outDimOri = {t_size * inputDims[1] * fzDim, inputDims[0] * fzDim};
  outputDesc.SetShape(GeShape(outDim));
  outputDesc.SetOriginShape(GeShape(outDimOri));
  outputDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_" + inputName, outputDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", t_size);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  for (int64_t i = 0; i < t_size; i++) {
    ge::GraphUtils::AddEdge(srcNodes[i]->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX[inputName]),
                            concatNode->GetInDataAnchor(t_size - 1 - i));  // Init_h
  }
  return concatNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddTConcatNode(const string& nodeName, const string& inputName,
                                                       vector<int64_t> fzDims, ge::NodePtr dynamicAUGRUGradNode,
                                                       vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                                       vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + nodeName, "ConcatD")),
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
  vector<int64_t> outDimOri = outputDesc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(outDim.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);
  outDim[0] = t_size;
  outDimOri[0] = t_size;
  outputDesc.SetShape(GeShape(outDim));
  outputDesc.SetOriginShape(GeShape(outDimOri));
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

ge::NodePtr DynamicAUGRUGradFusionPass::AddTConcatNodeBack(const string& nodeName, const string& inputName,
                                                           vector<int64_t> fzDims, ge::NodePtr dynamicAUGRUGradNode,
                                                           vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + nodeName, "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  FUSION_PASS_CHECK(srcNodes.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                     "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);

  GeTensorDesc inputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> inputDim = inputDesc.GetShape().GetDims();
  vector<int64_t> inputDimReshape = {inputDim[0], inputDim[1], inputDim[2], inputDim[3]};
  vector<int64_t> inputDimReshapeOri = {inputDim[1] * fzDim, inputDim[0] * fzDim};
  inputDesc.SetShape(GeShape(inputDimReshape));
  inputDesc.SetOriginShape(GeShape(inputDimReshapeOri));
  for (int64_t i = 0; i < t_size; i++) {
    concatDesc->AddInputDesc("input_" + to_string(i), inputDesc);
  }

  // output concat, shape:{t,batch_size,hidden_size}
  GeTensorDesc outputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> outDim = outputDesc.GetShape().GetDims();
  vector<int64_t> outDimNew = {outDim[0], t_size * outDim[1], outDim[2], outDim[3]};
  vector<int64_t> outDimNewOri = {t_size * outDim[1] * fzDim, outDim[0] * fzDim};
  outputDesc.SetShape(GeShape(outDimNew));
  outputDesc.SetOriginShape(GeShape(outDimNewOri));
  outputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  outputDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_" + inputName, outputDesc);

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

map<std::string, ge::NodePtr> DynamicAUGRUGradFusionPass::AddGRUHiddenGradNode(ge::NodePtr dynamicAUGRUGradNode,
                                                                               ge::ComputeGraph& graph,
                                                                               vector<ge::NodePtr>& newNodes,
                                                                               bool& failStatus) {
  map<std::string, ge::NodePtr> inputNodes;
  map<std::string, ge::NodePtr> result;
  vector<vector<ge::NodePtr>> result_node;
  if (t_size > 1) {
    // add loop t hidden grad nodes; [ [hidden_grad_nodes] [matmul_nodes] ]
    result_node = AddTLoopNode(inputNodes, dynamicAUGRUGradNode, graph, newNodes, failStatus);
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
    if (fusion_reduce) {
      vector<int64_t> fzDimsBack = {nzHiddenDim, nzBatch, 16, 16};
      dntXConcatTNode = AddTConcatNodeDnxBack("/AUGRUGrad/ConcatDntXBack", "dnt_x", fzDimsBack, dynamicAUGRUGradNode,
                                              result_node[0], graph, newNodes, failStatus);
      FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddDntXConcatTNode:check failed, fusion failed."),
                        return result);
    } else {
      dntXConcatTNode = AddTConcatNode("/AUGRUGrad/ConcatDntX", "dnt_x", fzDims, dynamicAUGRUGradNode,
                                       result_node[0], graph, newNodes, failStatus);
      FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddDntXConcatTNode:check failed, fusion failed."),
                        return result);
    }

    // add dgate_h concat node
    fzDims = {1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    ge::NodePtr dgateHConcatTNode = nullptr;
    if (fusion_reduce) {
      dgateHConcatTNode = AddTConcatNodeBack("/AUGRUGrad/ConcatDgateHBack", "dgate_h", fzDims, dynamicAUGRUGradNode,
                                             result_node[0], graph, newNodes, failStatus);
      FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddDgateHConcatTNode:check failed, fusion failed."),
                        return result);
    } else {
      dgateHConcatTNode = AddTConcatNode("/AUGRUGrad/ConcatDgateH", "dgate_h", fzDims, dynamicAUGRUGradNode,
                                         result_node[0], graph, newNodes, failStatus);
      FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddDgateHConcatTNode:check failed, fusion failed."),
                        return result);
    }

    // add dw_attr concat node
    ge::NodePtr dwAttConcatTNode = AddTConcatNode("/AUGRUGrad/ConcatDwAtt", "dw_att_t", fzDims, dynamicAUGRUGradNode,
                                                  result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "AddDwAttConcatTNode:check failed, fusion failed."),
    return result);

    result["dgate_h"] = dgateHConcatTNode;
    result["dnt_x"] = dntXConcatTNode;
    result["dw_att_t"] = dwAttConcatTNode;
  } else {
    result_node = AddTLoopNode(inputNodes, dynamicAUGRUGradNode, graph, newNodes, failStatus);
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
    result["dw_att_t"] = node;
  }
  ge::NodePtr dhPrevNode = result_node[0][result_node[0].size() - 1];
  result["dh_prev"] = dhPrevNode;
  return result;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddHTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph &graph,
                                                      vector<ge::NodePtr> &newNodes, bool &failStatus) {
  ge::OpDescPtr transDataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transDataDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUweightGrad/Dwh/HTransData",
                                                    "TransData")),
      transDataDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone();
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          transNode->GetInDataAnchor(0));
  return transNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddHSplitNode(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/SplitVD",
                                                "SplitVD")),
      splitDesc = nullptr;
      failStatus = true;
      return nullptr);

  // add transData
  ge::NodePtr transNode = AddHTransData(dynamicAUGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);
  // add input
  ge::GeTensorDesc inputTensorDescH = transNode->GetOpDesc()->GetOutputDesc(0).Clone();
  if (fusion_reduce) {
    inputTensorDescH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone();
    vector<int64_t> inputHDim = inputTensorDescH.GetShape().GetDims();
    FUSION_PASS_CHECK(inputHDim.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                        "AddTConcatNode:check failed, fusion failed."),
                      return nullptr);
    vector<int64_t> inputHReshapeDim = {inputHDim[0] * inputHDim[1], inputHDim[2]};
    inputTensorDescH.SetShape(GeShape(inputHReshapeDim));
    inputTensorDescH.SetOriginShape(GeShape(inputHReshapeDim));
  }
  splitDesc->AddInputDesc("input_h", inputTensorDescH);

  vector<int64_t> size_splits = {(t_size - 1) * batch, batch};
  if (fusion_reduce) {
    vector<int64_t> output1NzDim = {(t_size - 1) * batch, hidden_dim};
    AddOutputNodeDesc(splitDesc, "split_t_1", output1NzDim, ge::FORMAT_ND, output1NzDim, ge::FORMAT_ND,
                      inputHType);
    vector<int64_t> output2NzDim = {batch, hidden_dim};
    AddOutputNodeDesc(splitDesc, "split_1", output2NzDim, ge::FORMAT_ND, output2NzDim, ge::FORMAT_ND,
                      inputHType);
  } else {
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
  }
  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", splitSize);

  // create split node
  ge::NodePtr splitNode = AddNewNode(graph, splitDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  if (fusion_reduce) {
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                            splitNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), splitNode->GetInDataAnchor(0));
  }

  return splitNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddDwhTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                        vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr transDataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transDataDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUweightGrad/Dwh/TransData",
                                                    "TransData")),
      transDataDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          transNode->GetInDataAnchor(0));

  return transNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddHConcatNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr splitNode,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                       bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/HConcatD",
                                                 "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::NodePtr transNode = AddDwhTransData(dynamicAUGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);
  ge::GeTensorDesc inputTensorDescInitH = transNode->GetOpDesc()->GetOutputDesc(0).Clone();
  if (fusion_reduce) {
    inputTensorDescInitH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  }
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output concat_h, shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDim{t_size, batch, hidden_dim};
  vector<int64_t> outputNzDim{t_size, nzHiddenDim, nzBatch, 16, 16};
  if (fusion_reduce) {
    outputDim = {t_size * batch, hidden_dim};
    outputNzDim = {nzHiddenDim, t_size * nzBatch, 16, 16};
  }
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
  if (fusion_reduce) {
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                            concatNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  }
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  return concatNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr hConcatNode,
                                                         ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                         vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  if (fusion_reduce) {
    FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                                 "MatMulV2")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);
  } else {
    FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                                 "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);
  }

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
  if (fusion_reduce) {
    inputTensorDescDgate.SetOriginShape(GeShape({t_size * batch, (splitSize + 1) * hidden_dim}));
  } else {
    inputTensorDescDgate.SetOriginShape(GeShape({t_size, batch, (splitSize + 1) * hidden_dim}));
  }
  inputTensorDescDgate.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{t_size, hidden_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  if (fusion_reduce) {
    outputDim = {hidden_dim, (splitSize + 1) * hidden_dim};
    outputNzDim = {(splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  }
  AddOutputNodeDesc(matmulDesc, "dwt_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  if (fusion_reduce) {
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", true);
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", false);
  } else {
    ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
    ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);
  }

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

ge::NodePtr DynamicAUGRUGradFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr gruHiddenGradNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                                 "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
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
  vector<int64_t> outputDim{t_size, hidden_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  if (fusion_reduce) {
    outputDim = {t_size, hidden_dim, (splitSize + 1) * hidden_dim};
    outputNzDim = {t_size, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));  // Init_h
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            matmulNode->GetInDataAnchor(1));  // dgate_h
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_h
  }
  return matmulNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddDgateHSplitNodeBack(ge::NodePtr dynamicAUGRUGradNode,
                                                               ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                               vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/SplitVDBack",
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
  vector<int64_t> output1Dim{t_size * batch, splitSize * hidden_dim};
  vector<int64_t> output1NzDim{splitSize * nzHiddenDim, t_size * nzBatch, fzDim, fzDim};
  AddOutputNodeDesc(splitDesc, "split_ir", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1Dim, ge::FORMAT_ND,
                    inputHType);  // split_didr

  // add output2 split_1, shape:{t, batch, hidden_size}
  vector<int64_t> output2NzDim{nzHiddenDim, t_size * nzBatch, 16, 16};
  vector<int64_t> output2Dim = {t_size * batch, hidden_dim};
  AddOutputNodeDesc(splitDesc, "split_n", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_ND,
                    inputHType);  // split_dn_h

  // attr
  vector<int64_t> size_splits = {splitSize * nzHiddenDim * fzDim, nzHiddenDim * fzDim};
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

ge::NodePtr DynamicAUGRUGradFusionPass::AddDgateHSplitNode(ge::NodePtr dynamicAUGRUGradNode,
                                                           ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (splitDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/SplitVD",
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

ge::NodePtr DynamicAUGRUGradFusionPass::AddDgateXConcatNodeBack(ge::NodePtr dynamicAUGRUGradNode,
                                                                ge::NodePtr dgateHSplitNode,
                                                                ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                                vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD",
                                                 "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  vector<int64_t> dirtNzDesc = {splitSize * nzHiddenDim, t_size * nzBatch, fzDim, fzDim};
  vector<int64_t> dirNzDescOri = {t_size * nzBatch * fzDim, splitSize * nzHiddenDim * fzDim};
  vector<int64_t> dnxNzDesc = {nzHiddenDim, t_size * nzBatch, fzDim, fzDim};
  vector<int64_t> dnxDesc = {t_size * nzBatch * fzDim, nzHiddenDim * fzDim};
  AddInputNodeDesc(concatDesc, "input_dirt", dirtNzDesc, ge::FORMAT_FRACTAL_NZ, dirNzDescOri, ge::FORMAT_ND,
                   inputHType);
  AddInputNodeDesc(concatDesc, "input_dnt_x", dnxNzDesc, ge::FORMAT_FRACTAL_NZ, dnxDesc, ge::FORMAT_ND,
                   inputHType);

  // output shape:{t,batch,3*hidden_size}
  vector<int64_t> outputDim{t_size * batch, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{(splitSize + 1) * nzHiddenDim, t_size * nzBatch, fzDim, fzDim};
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

ge::NodePtr DynamicAUGRUGradFusionPass::AddDgateXConcatNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateHSplitNode,
                                                            ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD",
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
  vector<int64_t> outputDim{t_size, batch, (splitSize + 1) * hidden_dim};
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

Status DynamicAUGRUGradFusionPass::AddDxtMatmulNodeBack(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                        ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulOpDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmulV2",
                                                   "MatMulV2")), matmulOpDesc = nullptr;
  return false);

  // input
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dgate_x
  ge::GeTensorDesc weightXDesc =
      dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_input"]).Clone();  // weight_x

  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  weightXDesc.SetDataType(ge::DT_FLOAT16);
  weightXDesc.SetOriginFormat(ge::FORMAT_ND);
  weightXDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  weightXDesc.SetShape(GeShape({(splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim}));
  matmulOpDesc->AddInputDesc("dgate_x", dgateXDesc);
  matmulOpDesc->AddInputDesc("weight_x", weightXDesc);

  // add output dxt, shape:{t, batch, input_size}
  ge::GeTensorDesc outputTensorDesc = dynamicAUGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dx"]).Clone();
  outputTensorDesc.SetShape(GeShape({t_size * batch, nzInputDim * fzDim}));
  outputTensorDesc.SetOriginShape(GeShape({t_size * batch, nzInputDim * fzDim}));
  matmulOpDesc->AddOutputDesc("dxt", outputTensorDesc);

  // attr
  ge::AttrUtils::SetBool(matmulOpDesc, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulOpDesc, "transpose_x2", true);

  // create matmul node
  bool failStatus = false;
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulOpDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));  // dgate_x
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_input"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));

  // output Edge
  for (InDataAnchorPtr inAnchorPtr : dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
    // dxt
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return failStatus;
}

Status DynamicAUGRUGradFusionPass::AddDxtMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulOpDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmulV2",
                                                   "BatchMatMulV2")),
      matmulOpDesc = nullptr;
      return false);

  // input
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dgate_x
  ge::GeTensorDesc weightXDesc =
      dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_input"]).Clone();  // weight_x

  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  dgateXDesc.SetOriginFormat(ge::FORMAT_ND);
  dgateXDesc.SetOriginShape(GeShape({t_size, batch, (splitSize + 1) * hidden_dim}));
  weightXDesc.SetDataType(ge::DT_FLOAT16);
  weightXDesc.SetOriginFormat(ge::FORMAT_ND);
  weightXDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  weightXDesc.SetShape(GeShape({(splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim}));
  matmulOpDesc->AddInputDesc("dgate_x", dgateXDesc);
  matmulOpDesc->AddInputDesc("weight_x", weightXDesc);

  // add output dxt, shape:{t, batch, input_size}
  ge::GeTensorDesc outputTensorDesc = dynamicAUGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dx"]).Clone();
  matmulOpDesc->AddOutputDesc("dxt", outputTensorDesc);

  // attr
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x2", true);

  // create matmul node
  bool failStatus = false;
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulOpDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));  // dgate_x
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_input"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));

  // output Edge
  for (InDataAnchorPtr inAnchorPtr : dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
    // dxt
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return failStatus;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddDwxMatmulNodeBack(ge::NodePtr dynamicAUGRUGradNode,
                                                             ge::NodePtr dgateXConcatNode,
                                                             ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                             bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/BatchMatmul",
                                                 "MatMulV2")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc xtDesc = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).Clone();  // xt
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();              // dgate_x
  xtDesc.SetDataType(ge::DT_FLOAT16);
  xtDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  xtDesc.SetOriginFormat(ge::FORMAT_ND);
  xtDesc.SetShape(GeShape({nzInputDim, t_size * nzBatch, fzDim, fzDim}));
  xtDesc.SetOriginShape(GeShape({t_size * nzBatch * fzDim, nzInputDim * fzDim}));
  dgateXDesc.SetDataType(ge::DT_FLOAT16);

  matmulDesc->AddInputDesc("xt", xtDesc);
  matmulDesc->AddInputDesc("dgate_x", dgateXDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim{input_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{(splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim};
  AddOutputNodeDesc(matmulDesc, "dwt_x", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // input Edge
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));                                         // xt
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x
  return matmulNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddDwxMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/BatchMatmul",
                                                 "BatchMatMul")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc xtDesc = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).Clone();  // xt
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();              // dgate_x
  xtDesc.SetDataType(ge::DT_FLOAT16);
  xtDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  xtDesc.SetOriginFormat(ge::FORMAT_ND);
  xtDesc.SetShape(GeShape({t_size, nzInputDim, nzBatch, fzDim, fzDim}));
  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  dgateXDesc.SetOriginFormat(ge::FORMAT_ND);
  dgateXDesc.SetOriginShape(GeShape({t_size, batch, (splitSize + 1) * hidden_dim}));
  matmulDesc->AddInputDesc("xt", xtDesc);
  matmulDesc->AddInputDesc("dgate_x", dgateXDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim{t_size, input_dim, (splitSize + 1) * hidden_dim};
  vector<int64_t> outputNzDim{t_size, (splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim};
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));                                         // xt
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x
  return matmulNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                                         int anchorIndex, const vector<int64_t>& axis,
                                                         const string& nodeName, const string& indexName,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(
           dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD")),
      reduceSumDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  reduceSumDesc->AddInputDesc("input_" + nodeName, inputTensorDesc);

  // output
  ge::GeTensorDesc outputTensorDesc = dynamicAUGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX[indexName]).Clone();
  // no need to trans data while reduce 3 axis
  reduceSumDesc->AddOutputDesc("output_" + nodeName, outputTensorDesc);

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

  for (InDataAnchorPtr inAnchorPtr :
       dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddTReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                                          int anchorIndex,  const vector<int64_t>& axis,
                                                          const string& nodeName,
                                                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                          bool& failStatus) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(
           dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD")),
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
Status DynamicAUGRUGradFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dwxMatmulNode,
                                                      ge::NodePtr dwhMatmulNode, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& newNodes) {
  // add dw_x / dw_h reduce_sum
  if (t_size == 1) {
    // no need reduce_sum
    for (InDataAnchorPtr inAnchorPtr :
        dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_input"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dwxMatmulNode->GetOutDataAnchor(0), inAnchorPtr);
    }
    for (InDataAnchorPtr inAnchorPtr :
        dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_hidden"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dwhMatmulNode->GetOutDataAnchor(0), inAnchorPtr);
    }
    return SUCCESS;
  }
  int anchorOutputIndex = 0;
  vector<int64_t> reduceDwAxis{0};
  bool isFailure = false;
  AddReduceSumNode(dynamicAUGRUGradNode, dwxMatmulNode, anchorOutputIndex, reduceDwAxis, "dwx", "dw_input",
                   graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  if (fusion_reduce) {
    for (InDataAnchorPtr inAnchorPtr :
         dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_hidden"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dwhMatmulNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  } else {
    AddReduceSumNode(dynamicAUGRUGradNode, dwhMatmulNode, anchorOutputIndex, reduceDwAxis, "dwh", "dw_hidden",
                     graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwhReduceSumNode:check failed, fusion failed."),
                      return FAILED);
  }

  return SUCCESS;
}

ge::NodePtr DynamicAUGRUGradFusionPass::GetConstNodeOne(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                                        const vector<int64_t> &axis,
                                                        ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                                        bool &failStatus) {
  int64_t n_size = batch;
  ge::GeTensorPtr assitPtr = nullptr;
  int64_t matrixSize = (n_size + fzDim - 1) / fzDim * fzDim * fzDim * t_size;
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[matrixSize]());
  auto retMem = memset_s(inputAssit.get(), matrixSize, 1, matrixSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT("DynamicRnnGrad",
                                                                  "Failed to operate memset_s function."),
                    failStatus = true);
  float* dstConst = inputAssit.get();
  for (int j = 0; j < matrixSize; j++) {
    *(dstConst + j) = 1;
  }
  ge::GeTensorDesc tensorDesc;
  vector<int64_t> assit_dim_info = {};
  vector<int64_t> assit_dim_info_origin = {};
  if ((n_size % fzDim) != 0) {
    assit_dim_info = {t_size, (n_size + fzDim - 1) / fzDim, 1, fzDim, fzDim};
    assit_dim_info_origin = {t_size, 1, n_size};
  } else {
    assit_dim_info = {t_size * ((n_size + fzDim - 1) / fzDim), 1, fzDim, fzDim};
    assit_dim_info_origin = {1, n_size * t_size};
  }
  ge::GeShape assit_shape(assit_dim_info);
  ge::GeShape assit_shape_origin(assit_dim_info_origin);

  tensorDesc.SetShape(assit_shape);
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);
  tensorDesc.SetOriginShape(assit_shape_origin);
  tensorDesc.SetOriginDataType(ge::DT_FLOAT);
  FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 matrixSize * sizeof(float))),
       failStatus = true;
       return nullptr);

  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assitPtr);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);
  newNodes.push_back(const_node);

  return const_node;
}

ge::NodePtr DynamicAUGRUGradFusionPass::AddTMatMulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                                       ge::NodePtr constNode, const string& nodeName,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                       bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/"+nodeName+"/Matmul",
                                                 "MatMulV2")),
      matmulDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  ge::GeTensorDesc inputRightTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(0).Clone();  // xt
  ge::GeTensorDesc inputLeftTensorDesc = constNode->GetOpDesc()->GetOutputDesc(0).Clone();              // dgate_x
  inputLeftTensorDesc.SetDataType(ge::DT_FLOAT16);
  inputLeftTensorDesc.SetOriginDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_const", inputLeftTensorDesc);
  matmulDesc->AddInputDesc("input_dgate", inputRightTensorDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim = inputRightTensorDesc.GetShape().GetDims();
  vector<int64_t> matmulOutputDims = {outputDim[0], 1, fzDim, fzDim};
  vector<int64_t> matmulOutputOriDims = {1, outputDim[0] * fzDim};

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(GeShape(matmulOutputDims),
                                                       ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(GeShape(matmulOutputOriDims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // input Edge
  ge::GraphUtils::AddEdge(constNode->GetOutDataAnchor(0),
                          matmulNode->GetInDataAnchor(0)); // xt
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x

  return matmulNode;
}

Status DynamicAUGRUGradFusionPass::AddDbReduceSumNode(ge::NodePtr augruGradNode, ge::NodePtr dbxNode,
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
    AddReduceSumNode(augruGradNode, dbxNode, anchorOutputIndex, reduceDbxAxis, "dbx", "db_input",
                     graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbxReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    anchorOutputIndex = 1;
    AddReduceSumNode(augruGradNode, dbhNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden",
                     graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbhReduceSumNode:check failed, fusion failed."),
                      return FAILED);
    return SUCCESS;
  }
  // {t_size, 3 * nzHiddenDim, nzBatch, 16, 16}
  vector<int64_t> reduceDbTAxis{0};
  // dbx
  if (fusion_reduce) {
    ge::NodePtr const_one_node = GetConstNodeOne(augruGradNode, dbhNode,
                                                 reduceDbTAxis, graph, newNodes, isFailure);
    ge::NodePtr dbxTMatMulNode = AddTMatMulNode(augruGradNode, dbxNode, const_one_node,
                                                "dbx", graph, newNodes, isFailure);
    for (InDataAnchorPtr inAnchorPtr :
         augruGradNode->GetOutDataAnchor(OUTPUT_INDEX["db_input"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dbxTMatMulNode->GetOutDataAnchor(0), inAnchorPtr);
    }

    ge::NodePtr dbhTMatMulNode = AddTMatMulNode(augruGradNode, dbhNode, const_one_node,
                                                "dbh_t", graph, newNodes, isFailure);
    for (InDataAnchorPtr inAnchorPtr :
         augruGradNode->GetOutDataAnchor(OUTPUT_INDEX["db_hidden"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dbhTMatMulNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  } else {
    ge::NodePtr dbxTReduceSumNode = AddTReduceSumNode(augruGradNode, dbxNode, anchorOutputIndex, reduceDbTAxis,
                                                      "dbx_t", graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbxTReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    AddReduceSumNode(augruGradNode, dbxTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbx", "db_input", graph,
                     newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbxReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    // dbh
    ge::NodePtr dbhTReduceSumNode = AddTReduceSumNode(augruGradNode, dbhNode, anchorOutputIndex, reduceDbTAxis,
                                                      "dbh_t", graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbhTReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    AddReduceSumNode(augruGradNode, dbhTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden", graph,
                     newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbhReduceSumNode:check failed, fusion failed."),
                      return FAILED);
  }
  return SUCCESS;
}

Status DynamicAUGRUGradFusionPass::AddDwAttReduceSumNode(ge::NodePtr augruGradNode, ge::NodePtr dwAttNode,
                                                         ge::ComputeGraph &graph, vector <ge::NodePtr> &newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(augruGradNode->GetName() + "GRUWeightGrad/dwAtt/ReduceSumD",
                                                    "ReduceSumD")), reduceSumDesc = nullptr; return FAILED);

  // input
  vector<int64_t> dwAttInshape = {t_size, batch, nzHiddenDim * fzDim};
  AddInputNodeDesc(reduceSumDesc, "x", dwAttInshape, ge::FORMAT_ND, dwAttInshape, ge::FORMAT_ND,
                   inputHType);

  // output
  vector<int64_t> dwAttOutshape = {t_size, batch};
  AddOutputNodeDesc(reduceSumDesc, "y", dwAttOutshape, ge::FORMAT_ND, dwAttOutshape,
                    ge::FORMAT_ND, inputHType);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {2,});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  bool failStatus = false;
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return FAILED);

  // Edge
  int anchorOutputIndex = 0;
  if (t_size == 1) {
    anchorOutputIndex = 3;
  }
  ge::GraphUtils::AddEdge(dwAttNode->GetOutDataAnchor(anchorOutputIndex), reduceSumNode->GetInDataAnchor(0));

  for (InDataAnchorPtr inAnchorPtr :
      augruGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_att"])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return SUCCESS;
}

Status DynamicAUGRUGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicAUGRUGradFusionPass fusion begin.");
  bool isFailure = false;
  // get dynamicAUGRUGradNode
  ge::NodePtr augruGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(augruGradNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicAUGRUGrad:grad node is null, fusion failed."),
                    return FAILED);
  ge::OpDescPtr dynamicAUGRUGradDesc = augruGradNode->GetOpDesc();
  FUSION_PASS_CHECK(dynamicAUGRUGradDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "DynamicAUGRUGrad:op desc is null, fusion failed."),
                    return FAILED);
  ge::GeTensorDesc inputTensorDescH = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  ge::GeTensorDesc inputTensorDescX = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  batch = inputTensorDescH.GetShape().GetDim(1);
  hidden_dim = inputTensorDescH.GetShape().GetDim(splitSize);
  input_dim = inputTensorDescX.GetShape().GetDim(splitSize);

  t_size = inputTensorDescH.GetShape().GetDim(0);
  if (batch % fzDim == 0 && hidden_dim % fzDim == 0 && input_dim % fzDim == 0 && t_size != 1) {
    fusion_reduce = true;
  }
  if (hidden_dim % 16 != 0 || input_dim % 16 != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is not 16 align, will not changed");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(batch) ||
      PatternFusionUtil::IsUnknownShape(hidden_dim) || PatternFusionUtil::IsUnknownShape(t_size) ||
      PatternFusionUtil::IsUnknownShape(input_dim)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "DynamicAUGRUGradFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  // init shape
  this->GetNodeInfo(augruGradNode);

  // add gruHiddenGrad {dhPrevNode, dgateHConcatTNode, dntXConcatTNode}
  map<std::string, ge::NodePtr> hiddenGradNodes = AddGRUHiddenGradNode(augruGradNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddGRUHiddenGradNode:check failed, fusion failed."),
                    return FAILED);

  ge::NodePtr dwhMatmulNode;
  if (t_size != 1) {
    // add split
    ge::NodePtr splitNode = AddHSplitNode(augruGradNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddHSplitNode:check failed, fusion failed."),
                      return FAILED);

    // add concat
    ge::NodePtr hConcatNode = AddHConcatNode(augruGradNode, splitNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddHConcatNode:check failed, fusion failed."),
                      return FAILED);

    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode =
        AddDwhMatmulNode(augruGradNode, hConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  } else {
    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode = AddDwhMatmulNode(augruGradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  }

  // split dgate_h to [dit, drt] and [dnt_h]
  ge::NodePtr dgateHSplitNode = nullptr;
  if (fusion_reduce) {
    dgateHSplitNode = AddDgateHSplitNodeBack(augruGradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  } else {
    dgateHSplitNode = AddDgateHSplitNode(augruGradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  }
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // concat [dit, drt] with [dnt_x] to dgate_x
  ge::NodePtr gateConcatNode = nullptr;
  if (fusion_reduce) {
    gateConcatNode = AddDgateXConcatNodeBack(augruGradNode,
                                             dgateHSplitNode, hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);
  } else {
    gateConcatNode = AddDgateXConcatNode(augruGradNode, dgateHSplitNode,
                                         hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);
  }
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateXConcatNode:check failed, fusion failed."),
                    return FAILED);

  // add dxt matmul(dgate_x, w_x.T)
  if (fusion_reduce) {
    isFailure = AddDxtMatmulNodeBack(augruGradNode, gateConcatNode, graph, newNodes);
  } else {
    isFailure = AddDxtMatmulNode(augruGradNode, gateConcatNode, graph, newNodes);
  }
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDxtMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x matmul(x.T, dgate_x)
  ge::NodePtr dwxMatmulNode = nullptr;
  if (fusion_reduce) {
    dwxMatmulNode = AddDwxMatmulNodeBack(augruGradNode, gateConcatNode, graph, newNodes, isFailure);
  } else {
    dwxMatmulNode = AddDwxMatmulNode(augruGradNode, gateConcatNode, graph, newNodes, isFailure);
  }
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxMatmulNode:check failed, fusion failed."),
                    return FAILED);
  if (fusion_reduce) {
    for (InDataAnchorPtr inAnchorPtr :
         augruGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_hidden"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dwhMatmulNode->GetOutDataAnchor(0), inAnchorPtr);
    }

    for (InDataAnchorPtr inAnchorPtr :
         augruGradNode->GetOutDataAnchor(OUTPUT_INDEX["dw_input"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(dwxMatmulNode->GetOutDataAnchor(0), inAnchorPtr);
    }
    isFailure = AddDbReduceSumNode(augruGradNode, gateConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes);
  } else {
    isFailure = AddDwReduceSumNode(augruGradNode, dwxMatmulNode, dwhMatmulNode, graph, newNodes);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDwReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    isFailure = AddDbReduceSumNode(augruGradNode, gateConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes);
  }
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbReduceSumNode:check failed, fusion failed."),
                    return FAILED);
  isFailure = AddDwAttReduceSumNode(augruGradNode, hiddenGradNodes["dw_att_t"], graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwAttReduceSumNode:check failed, fusion failed."),
  return FAILED);

  // unlink all control input of augruGradNode
  if (augruGradNode->GetInControlAnchor() != nullptr) {
    augruGradNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of augruGradNode
  for (auto inAnchor : augruGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove augruGradNode from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(augruGradNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                     augruGradNode->GetName().c_str()),
      return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicAUGRUGradFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("DynamicAUGRUGradFusionPass", BUILT_IN_GRAPH_PASS, DynamicAUGRUGradFusionPass);
}  // namespace fe
