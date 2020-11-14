/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2Grad fusion pass(DynamicGRUV2Grad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_gru_v2_grad_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

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

vector<FusionPattern*> DynamicGRUV2GradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicGRUV2GradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicGRUV2GradFusionPass::GetNodeInfo(ge::NodePtr dynamicGRUGradNode) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  t_size = inputTensorDescH.GetShape().GetDim(0);
  batch = inputTensorDescH.GetShape().GetDim(1);
  nzBatch = (batch + 15) / 16;
  hidden_dim = inputTensorDescH.GetShape().GetDim(2);
  nzHiddenDim = (hidden_dim + 15) / 16;

  ge::GeTensorDesc inputTensorDescX = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  input_dim = inputTensorDescX.GetShape().GetDim(2);
  nzInputDim = (input_dim + 15) / 16;
  inputHType = inputTensorDescH.GetDataType();
  return;
}

void DynamicGRUV2GradFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

void DynamicGRUV2GradFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

void DynamicGRUV2GradFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

ge::NodePtr DynamicGRUV2GradFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc,
                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::NodePtr node = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", node->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(node);
  return node;
}

void DynamicGRUV2GradFusionPass::AddHiddenGradNodeEdge(map<std::string, ge::NodePtr>& inputNodes,
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
    if (t_size == 1) {
      ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                              hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
    } else {
      ge::GraphUtils::AddEdge(inputNodes["h"]->GetOutDataAnchor(t_size - curT - 2),
                              hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
    }
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
    // fake connect the last cell
    curT = t_size - 1;
  }

  if (t_size == 1) {
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
  } else {
    ge::GraphUtils::AddEdge(inputNodes["dy"]->GetOutDataAnchor(t_size - curT - 1),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dy"]));
    ge::GraphUtils::AddEdge(inputNodes["update"]->GetOutDataAnchor(t_size - curT - 1),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["update"]));
    ge::GraphUtils::AddEdge(inputNodes["reset"]->GetOutDataAnchor(t_size - curT - 1),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["reset"]));
    ge::GraphUtils::AddEdge(inputNodes["new"]->GetOutDataAnchor(t_size - curT - 1),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["new"]));
    ge::GraphUtils::AddEdge(inputNodes["hidden_new"]->GetOutDataAnchor(t_size - curT - 1),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["hidden_new"]));
  }
}

void DynamicGRUV2GradFusionPass::AddSplitTInputNodeDesc(ge::OpDescPtr hiddenGradDesc, ge::OpDescPtr dynamicGRUGradDesc,
                                                        const std::string& name) {
  ge::GeTensorDesc inputDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX[name]).Clone();
  // change dim t to 1
  vector<int64_t> dims = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "AddSplitTInputNodeDesc:check failed, fusion failed."),
                    return );
  dims[0] = 1;
  vector<int64_t> originDims = inputDesc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(originDims.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "AddSplitTInputNodeDesc:check failed, fusion failed."), return );
  originDims[0] = 1;
  inputDesc.SetShape(ge::GeShape(dims));
  inputDesc.SetOriginShape(ge::GeShape(originDims));
  hiddenGradDesc->AddInputDesc(name, inputDesc);
  return;
}

int64_t DynamicGRUV2GradFusionPass::GetTState(int64_t curT) {
  if (curT == 0) {
    if (t_size == 1) {
      return 0;
    } else {
      return 1;
    }
  }
  if (curT < t_size - 1) {
    return 2;
  }
  if (curT == t_size - 1) {
    return 3;
  }
  return 4;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddOneHiddenGradNode(const string& gateOrder, int64_t curT,
                                                             ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();

  ge::OpDescPtr hiddenGradDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "/GRUV2Grad/GRUV2HiddenGradCell_" + std::to_string(curT), "GRUV2HiddenGradCell");

  // set attr of gate order
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);
  // set attr of t_state
  ge::AttrUtils::SetInt(hiddenGradDesc, "t_state", GetTState(curT));

  // set input desc
  ge::GeTensorDesc dhPrevDesc = dynamicGRUGradDesc->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).Clone();
  hiddenGradDesc->AddInputDesc("dh_pre_t", dhPrevDesc);
  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "h");
  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "dy");
  ge::GeTensorDesc dhDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dh"]).Clone();
  if (curT == 0) {
    hiddenGradDesc->AddInputDesc("dh", dhDesc);
  } else {
    AddInputNodeDesc(hiddenGradDesc, "dh", {1, nzHiddenDim, nzBatch, 16, 16}, ge::FORMAT_FRACTAL_NZ,
                     {1, hidden_dim, batch}, ge::FORMAT_ND, inputHType);
  }

  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "update");
  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "reset");
  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "new");
  AddSplitTInputNodeDesc(hiddenGradDesc, dynamicGRUGradDesc, "hidden_new");

  vector<int64_t> dgateHDim{1, batch, 3 * hidden_dim};
  vector<int64_t> dgateHNzDim{1, 3 * nzHiddenDim, nzBatch, 16, 16};
  vector<int64_t> singleGateDim{1, batch, hidden_dim};
  vector<int64_t> singleGateNzDim{1, nzHiddenDim, nzBatch, 16, 16};

  hiddenGradDesc->AddOutputDesc("dh_prev", dhPrevDesc);
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHNzDim, ge::FORMAT_FRACTAL_NZ, dgateHNzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateNzDim, ge::FORMAT_FRACTAL_NZ, singleGateNzDim,
                    ge::FORMAT_FRACTAL_NZ, inputHType);

  // create gru_hidden_grad node
  ge::NodePtr hiddenGradNode = this->AddNewNode(graph, hiddenGradDesc, newNodes, failStatus);
  return hiddenGradNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode,
                                                                   ge::NodePtr dynamicGRUGradNode,
                                                                   ge::ComputeGraph& graph,
                                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "/GRUV2Grad/Matmul_" + to_string(curT), "BatchMatMul");
  // input
  ge::GeTensorDesc inputDesc = hiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  inputDesc.SetDataType(ge::DT_FLOAT16);
  vector<int64_t> inputDim{1, batch, 3 * hidden_dim};
  vector<int64_t> inputNzDim{1, 3 * nzHiddenDim, nzBatch, 16, 16};
  inputDesc.SetOriginShape(GeShape(inputDim));
  inputDesc.SetShape(GeShape(inputNzDim));
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);

  // weight
  ge::GeTensorDesc weightDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone();
  weightDesc.SetOriginFormat(ge::FORMAT_ND);
  weightDesc.SetOriginShape(GeShape({1, hidden_dim, 3 * hidden_dim}));
  weightDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  vector<int64_t> weightNzDim = {1, 3 * nzHiddenDim, nzHiddenDim, 16, 16};
  weightDesc.SetShape(GeShape(weightNzDim));
  weightDesc.SetDataType(ge::DT_FLOAT16);

  matmulDesc->AddInputDesc("input_dgate_h", inputDesc);
  matmulDesc->AddInputDesc("input_weight", weightDesc);

  vector<int64_t> outputDim{1, batch, hidden_dim};
  vector<int64_t> outputNzDim{1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(matmulDesc, "dh", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, ge::DT_FLOAT);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", true);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                          matmulNode->GetInDataAnchor(0));  // dgate_h
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));  // weight
  return matmulNode;
}

vector<vector<ge::NodePtr>> DynamicGRUV2GradFusionPass::AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes,
                                                                     ge::NodePtr dynamicGRUGradNode,
                                                                     ge::ComputeGraph& graph,
                                                                     vector<ge::NodePtr>& newNodes, bool& failStatus) {
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
    ge::NodePtr matmulNode =
        AddOneHiddenGradMatmulNode(i, hiddenGradNode, dynamicGRUGradNode, graph, newNodes, failStatus);
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
  AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, nullptr, lastHiddenGradNode, lastMatmulNode, dynamicGRUGradNode,
                        t_size);
  hiddenGradNodes.push_back(hiddenGradNode);

  result.push_back(hiddenGradNodes);
  result.push_back(matmulNodes);
  return result;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddTSplitNode(const string& nodeName, const string& inputName,
                                                      ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + nodeName, "SplitVD");

  // add input
  ge::GeTensorDesc inputTensorDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX[inputName]).Clone();
  splitDesc->AddInputDesc("input_" + inputName, inputTensorDesc);
  vector<int64_t> outputDim = inputTensorDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(outputDim.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "AddTSplitNode:check failed, fusion failed."),
                    return nullptr);
  outputDim[0] = 1;

  // add output1 split_t_1, shape:{1,batch_size,hidden_size}
  vector<int64_t> sizeSplits = {};
  for (int64_t i = 0; i < this->t_size; i++) {
    AddOutputNodeDesc(splitDesc, "split_t_" + std::to_string(i), outputDim, inputHType, ge::FORMAT_ND);
    sizeSplits.push_back(1);
  }
  // attr
  ge::AttrUtils::SetListInt(splitDesc, "size_splits", sizeSplits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", this->t_size);

  // create split node
  ge::NodePtr splitNode = this->AddNewNode(graph, splitDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX[inputName])->GetPeerOutAnchor(),
                          splitNode->GetInDataAnchor(0));
  return splitNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddTConcatNode(const string& nodeName, const string& inputName,
                                                       vector<int64_t> fzDims, ge::NodePtr dynamicGRUGradNode,
                                                       vector<ge::NodePtr>& srcNodes, ge::ComputeGraph& graph,
                                                       vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + nodeName, "ConcatD");
  // input
  FUSION_PASS_CHECK(srcNodes.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);

  GeTensorDesc inputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  for (int64_t i = 0; i < t_size; i++) {
    concatDesc->AddInputDesc("input_" + to_string(i), inputDesc);
  }

  // output concat, shape:{t,batch_size,hidden_size}
  GeTensorDesc outputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> outDim = outputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(outDim.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "AddTConcatNode:check failed, fusion failed."),
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

  // Edge
  for (int64_t i = 0; i < t_size; i++) {
    ge::GraphUtils::AddEdge(srcNodes[i]->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX[inputName]),
                            concatNode->GetInDataAnchor(t_size - 1 - i));  // Init_h
  }
  return concatNode;
}

map<std::string, ge::NodePtr> DynamicGRUV2GradFusionPass::AddGRUHiddenGradNode(ge::NodePtr dynamicGRUGradNode,
                                                                               ge::ComputeGraph& graph,
                                                                               vector<ge::NodePtr>& newNodes,
                                                                               bool& failStatus) {
  map<std::string, ge::NodePtr> inputNodes;
  map<std::string, ge::NodePtr> result;
  vector<vector<ge::NodePtr>> result_node;
  if (t_size > 1) {
    // add h split node
    ge::NodePtr hSplitNode = AddTSplitNode("/GRUV2Grad/SplitH", "h", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddH2SplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["h"] = hSplitNode;
    // add dy split node
    ge::NodePtr dySplitNode = AddTSplitNode("/GRUV2Grad/SplitY", "dy", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDySplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["dy"] = dySplitNode;
    // add update split node
    ge::NodePtr updateSplitNode =
        AddTSplitNode("/GRUV2Grad/SplitUpdate", "update", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddUpdateSplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["update"] = updateSplitNode;
    // add reset split node
    ge::NodePtr resetSplitNode =
        AddTSplitNode("/GRUV2Grad/SplitReset", "reset", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddResetSplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["reset"] = resetSplitNode;
    // add new split node
    ge::NodePtr newSplitNode =
        AddTSplitNode("/GRUV2Grad/SplitNew", "new", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddNewSplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["new"] = newSplitNode;
    // add hidden_new node
    ge::NodePtr hiddenNewSplit =
        AddTSplitNode("/GRUV2Grad/SplitHiddenNew", "hidden_new", dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHiddenNewSplitNode:check failed, fusion failed."),
                      return result);
    inputNodes["hidden_new"] = hiddenNewSplit;

    // add loop t hidden grad nodes; [ [hidden_grad_nodes] [matmul_nodes] ]
    result_node = AddTLoopNode(inputNodes, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(result_node.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                      return result);
    FUSION_PASS_CHECK(result_node[0].empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                      return result);

    // add dnt_x concat node by t
    vector<int64_t> fzDims = {1, nzHiddenDim, nzBatch, 16, 16};
    ge::NodePtr dntXConcatTNode = AddTConcatNode("/GRUV2Grad/ConcatDntX", "dnt_x", fzDims, dynamicGRUGradNode,
                                                 result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDntXConcatTNode:check failed, fusion failed."),
                      return result);

    // add dgate_h concat node
    fzDims = {1, 3 * nzHiddenDim, nzBatch, 16, 16};
    ge::NodePtr dgateHConcatTNode = AddTConcatNode("/GRUV2Grad/ConcatDgateH", "dgate_h", fzDims, dynamicGRUGradNode,
                                                   result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDgateHConcatTNode:check failed, fusion failed."),
                      return result);
    result["dgate_h"] = dgateHConcatTNode;
    result["dnt_x"] = dntXConcatTNode;
  } else {
    result_node = AddTLoopNode(inputNodes, dynamicGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(result_node.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                      return result);
    FUSION_PASS_CHECK(result_node[0].empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                      return result);
    ge::NodePtr node = result_node[0][0];
    result["dgate_h"] = node;
    result["dnt_x"] = node;
  }
  ge::NodePtr dhPrevNode = result_node[0][result_node[0].size() - 1];
  result["dh_prev"] = dhPrevNode;
  return result;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/SplitVD", "SplitVD");

  // add input
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]);
  inputTensorDescH.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescH.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescH.SetShape(GeShape({t_size, nzHiddenDim, nzBatch, 16, 16}));
  inputTensorDescH.SetOriginShape(GeShape({t_size, nzHiddenDim, nzBatch, 16, 16}));
  splitDesc->AddInputDesc("input_h", inputTensorDescH);

  // add output1 split_t_1, shape:{t-1,batch_size,hidden_size}
  vector<int64_t> output1Dim{t_size - 1, batch, hidden_dim};
  vector<int64_t> output1NzDim{t_size - 1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_t_1", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_t_1

  // add output2 split_1, shape:{1,batch_size,hidden_size}
  vector<int64_t> output2Dim{1, batch, hidden_dim};
  vector<int64_t> output2NzDim{1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_1", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_1

  // attr
  vector<int64_t> size_splits{t_size - 1, 1};
  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);

  // create split node
  ge::NodePtr splitNode = AddNewNode(graph, splitDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          splitNode->GetInDataAnchor(0));
  return splitNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddDwhTransData(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                        vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr transDataDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUweightGrad/Dwh/TransData", "TransData");
  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  inputTensorDescInitH.SetShape(GeShape({1, batch, hidden_dim}));
  transDataDesc->AddInputDesc("trans_src", inputTensorDescInitH);
  // output
  vector<int64_t> dstDims = {1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(transDataDesc, "trans_dst", dstDims, inputHType, ge::FORMAT_FRACTAL_NZ);

  // attr
  ge::AttrUtils::SetStr(transDataDesc, "src_format", "ND");
  ge::AttrUtils::SetStr(transDataDesc, "dst_format", "FRACTAL_NZ");

  // create node
  ge::NodePtr transNode = AddNewNode(graph, transDataDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          transNode->GetInDataAnchor(0));

  return transNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddHConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr splitNode,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                       bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/HConcatD", "ConcatD");
  // input
  ge::NodePtr transNode = AddDwhTransData(dynamicGRUGradNode, graph, newNodes, failStatus);
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

  // Edge
  ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  return concatNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hConcatNode,
                                                         ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                         vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul", "BatchMatMul");

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
  inputTensorDescDgate.SetOriginShape(GeShape({t_size, batch, 3 * hidden_dim}));
  inputTensorDescDgate.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{t_size, hidden_dim, 3 * hidden_dim};
  vector<int64_t> outputNzDim{t_size, 3 * nzHiddenDim, nzHiddenDim, 16, 16};
  AddOutputNodeDesc(matmulDesc, "dwt_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);

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

ge::NodePtr DynamicGRUV2GradFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr gruHiddenGradNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul", "BatchMatMul");

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

  inputTensorDescDgate.SetShape(GeShape({1, 3 * nzHiddenDim, nzBatch, 16, 16}));
  inputTensorDescDgate.SetOriginShape(GeShape({1, batch, 3 * hidden_dim}));
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{t_size, hidden_dim, 3 * hidden_dim};
  vector<int64_t> outputNzDim{t_size, 3 * nzHiddenDim, nzHiddenDim, 16, 16};
  AddOutputNodeDesc(matmulDesc, "dwt_h", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulDesc, newNodes, failStatus);

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

ge::NodePtr DynamicGRUV2GradFusionPass::AddDgateHSplitNode(ge::NodePtr dynamicGRUGradNode,
                                                           ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create split desc
  ge::OpDescPtr splitDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/SplitVD", "SplitVD");

  // add input
  ge::GeTensorDesc inputDgateH;
  if (t_size == 1) {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  } else {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }
  splitDesc->AddInputDesc("input_dgate_h", inputDgateH);

  // add output1 dgate_ir, shape:{t, batch, 2 * hidden_size}
  vector<int64_t> output1Dim{t_size, batch, 2 * hidden_dim};
  vector<int64_t> output1NzDim{t_size, 2 * nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_ir", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_didr

  // add output2 split_1, shape:{t, batch, hidden_size}
  vector<int64_t> output2NzDim{t_size, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_n", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_dn_h

  // attr
  vector<int64_t> size_splits = {2 * nzHiddenDim, nzHiddenDim};
  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 1);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);

  // create split node
  ge::NodePtr splitNode = AddNewNode(graph, splitDesc, newNodes, failStatus);

  // Edge
  if (t_size == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            splitNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), splitNode->GetInDataAnchor(0));
  }
  return splitNode;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddDgateXConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateHSplitNode,
                                                            ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD", "ConcatD");

  // input
  vector<int64_t> dirtNzDesc = {t_size, 2 * nzHiddenDim, nzBatch, 16, 16};
  vector<int64_t> dnxNzDesc = {t_size, nzHiddenDim, nzBatch, 16, 16};
  AddInputNodeDesc(concatDesc, "input_dirt", dirtNzDesc, ge::FORMAT_FRACTAL_NZ, dirtNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);
  AddInputNodeDesc(concatDesc, "input_dnt_x", dnxNzDesc, ge::FORMAT_FRACTAL_NZ, dnxNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);

  // output shape:{t,batch,3*hidden_size}
  vector<int64_t> outputDim{t_size, batch, 3 * hidden_dim};
  vector<int64_t> outputNzDim{t_size, 3 * nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(concatDesc, "dgate_x", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 1);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);

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

ge::NodePtr DynamicGRUV2GradFusionPass::AddWxBroadcastNode(ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create broadcast desc
  ge::OpDescPtr broadcastDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dx/Broadcast", "BroadcastToD");

  // input
  ge::GeTensorDesc inputWxDesc =
      dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_input"]).Clone();  // weight_x
  vector<int64_t> inputWx{1, input_dim, 3 * hidden_dim};
  ge::GeShape inputWxShape(inputWx);
  inputWxDesc.SetShape(inputWxShape);
  inputWxDesc.SetOriginShape(inputWxShape);
  broadcastDesc->AddInputDesc("wx_2d", inputWxDesc);

  // output shape:{t, input_size, 3*hidden_size}
  vector<int64_t> outputDim{t_size, input_dim, 3 * hidden_dim};
  AddOutputNodeDesc(broadcastDesc, "wx_3d", outputDim, inputWxDesc.GetDataType(), ge::FORMAT_ND);

  // attr
  ge::AttrUtils::SetListInt(broadcastDesc, "shape", outputDim);

  // create broadcast node
  ge::NodePtr broadcastNode = AddNewNode(graph, broadcastDesc, newNodes, failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_input"])->GetPeerOutAnchor(),
                          broadcastNode->GetInDataAnchor(0));  // dgate_x
  return broadcastNode;
}

Status DynamicGRUV2GradFusionPass::AddDxtMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                    ge::NodePtr wxBoadcastNode, ge::ComputeGraph& graph,
                                                    vector<ge::NodePtr>& newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmul", "BatchMatMul");

  // input
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dgate_x
  ge::GeTensorDesc weightXDesc = wxBoadcastNode->GetOpDesc()->GetOutputDesc(0).Clone();   // weight_x
  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  dgateXDesc.SetOriginFormat(ge::FORMAT_ND);
  dgateXDesc.SetOriginShape(GeShape({t_size, batch, 3 * hidden_dim}));
  weightXDesc.SetDataType(ge::DT_FLOAT16);
  weightXDesc.SetOriginFormat(ge::FORMAT_ND);
  weightXDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  weightXDesc.SetShape(GeShape({t_size, 3 * nzHiddenDim, nzInputDim, 16, 16}));
  matmulOpDesc->AddInputDesc("dgate_x", dgateXDesc);
  matmulOpDesc->AddInputDesc("weight_x", weightXDesc);

  // add output dxt, shape:{t, batch, input_size}
  ge::GeTensorDesc outputTensorDesc = dynamicGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dx"]).Clone();  // dw
  matmulOpDesc->AddOutputDesc("dxt", outputTensorDesc);

  // attr
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x2", true);

  // create matmul node
  bool failStatus = false;
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulOpDesc, newNodes, failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));  // dgate_x
  ge::GraphUtils::AddEdge(wxBoadcastNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));    // w_x_3d

  // output Edge
  for (InDataAnchorPtr inAnchorPtr : dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
    // dxt
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return failStatus;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddDwxMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/BatchMatmul", "BatchMatMul");

  // input
  ge::GeTensorDesc xtDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).Clone();  // xt
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();              // dgate_x
  xtDesc.SetDataType(ge::DT_FLOAT16);
  xtDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  xtDesc.SetOriginFormat(ge::FORMAT_ND);
  xtDesc.SetShape(GeShape({t_size, nzInputDim, nzBatch, 16, 16}));
  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  dgateXDesc.SetOriginFormat(ge::FORMAT_ND);
  dgateXDesc.SetOriginShape(GeShape({t_size, batch, 3 * hidden_dim}));
  matmulDesc->AddInputDesc("xt", xtDesc);
  matmulDesc->AddInputDesc("dgate_x", dgateXDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim{t_size, input_dim, 3 * hidden_dim};
  vector<int64_t> outputNzDim{t_size, 3 * nzHiddenDim, nzInputDim, 16, 16};
  AddOutputNodeDesc(matmulDesc, "dwt_x", outputNzDim, ge::FORMAT_FRACTAL_NZ, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);

  // input Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));                                         // xt
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x
  return matmulNode;
}

Status DynamicGRUV2GradFusionPass::AddReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode,
                                                    int anchorIndex, const vector<int64_t>& axis,
                                                    const string& nodeName, const string& indexName,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/ReduceSumD", "ReduceSumD");

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  inputTensorDesc.SetFormat(ge::FORMAT_ND);
  inputTensorDesc.SetShape(inputTensorDesc.GetOriginShape());
  reduceSumDesc->AddInputDesc("input_" + nodeName, inputTensorDesc);

  // output
  ge::GeTensorDesc outputTensorDesc = dynamicGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX[indexName]).Clone();
  reduceSumDesc->AddOutputDesc("output_" + nodeName, outputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", axis);
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  bool failStatus = false;
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), reduceSumNode->GetInDataAnchor(0));

  for (InDataAnchorPtr inAnchorPtr :
       dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return SUCCESS;
}

ge::NodePtr DynamicGRUV2GradFusionPass::AddNzTransDataNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode,
                                                           int anchorIndex, const string& nodeName,
                                                           ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                           bool& failStatus) {
  // create desc
  ge::OpDescPtr transDataDesc = std::make_shared<ge::OpDesc>(
      dynamicGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransData", "TransData");

  // input
  ge::GeTensorDesc inputTensorDesc = inputNode->GetOpDesc()->GetOutputDesc(anchorIndex).Clone();
  transDataDesc->AddInputDesc("trans_src", inputTensorDesc);

  vector<int64_t> dstDims = {t_size, batch, 3 * hidden_dim};
  AddOutputNodeDesc(transDataDesc, "trans_dst", dstDims, inputHType, ge::FORMAT_ND);

  // attr
  ge::AttrUtils::SetStr(transDataDesc, "src_format", "FRACTAL_NZ");
  ge::AttrUtils::SetStr(transDataDesc, "dst_format", "ND");

  // create node
  ge::NodePtr transNode = AddNewNode(graph, transDataDesc, newNodes, failStatus);

  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), transNode->GetInDataAnchor(0));
  return transNode;
}

Status DynamicGRUV2GradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradFusionPass fusion begin.");
  bool isFailure = false;
  // get dynamicGRUGradNode
  ge::NodePtr gruV2GradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  // init shape
  this->GetNodeInfo(gruV2GradNode);

  // add gruHiddenGrad {dhPrevNode, dgateHConcatTNode, dntXConcatTNode}
  map<std::string, ge::NodePtr> hiddenGradNodes = AddGRUHiddenGradNode(gruV2GradNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddGRUHiddenGradNode:check failed, fusion failed."),
                    return FAILED);

  ge::NodePtr dwhMatmulNode;
  if (t_size != 1) {
    // add split
    ge::NodePtr splitNode = AddHSplitNode(gruV2GradNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHSplitNode:check failed, fusion failed."),
                      return FAILED);

    // add concat
    ge::NodePtr hConcatNode = AddHConcatNode(gruV2GradNode, splitNode, graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHConcatNode:check failed, fusion failed."),
                      return FAILED);

    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode =
        AddDwhMatmulNode(gruV2GradNode, hConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  } else {
    // add dw_h : matmul(h_prev.T, dgate_h)
    dwhMatmulNode = AddDwhMatmulNode(gruV2GradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
    FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwhMatmulNode:check failed, fusion failed."),
                      return FAILED);
  }

  // split dgate_h to [dit, drt] and [dnt_h]
  ge::NodePtr dgateHSplitNode =
      AddDgateHSplitNode(gruV2GradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDgateHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // concat [dit, drt] with [dnt_x] to dgate_x
  ge::NodePtr gateConcatNode =
      AddDgateXConcatNode(gruV2GradNode, dgateHSplitNode, hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDgateXConcatNode:check failed, fusion failed."),
                    return FAILED);

  // broadcast wx from [input_dim, 3*hidden_dim] to  [input_dim, 3*hidden_dim]
  ge::NodePtr wxBroadcastNode = AddWxBroadcastNode(gruV2GradNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddWxBroadcastNode:check failed, fusion failed."),
                    return FAILED);

  // add dxt matmul(dgate_x, w_x.T)
  isFailure = AddDxtMatmulNode(gruV2GradNode, gateConcatNode, wxBroadcastNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDxtMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x matmul(x.T, dgate_x)
  ge::NodePtr dwxMatmulNode = AddDwxMatmulNode(gruV2GradNode, gateConcatNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwxMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x / dw_h reduce_sum
  vector<int64_t> reduceDwAxis{0};
  int anchorOutputIndex = 0;
  isFailure = AddReduceSumNode(gruV2GradNode, dwxMatmulNode, anchorOutputIndex, reduceDwAxis, "dwx", "dw_input", graph,
                               newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddReduceSumNode(gruV2GradNode, dwhMatmulNode, anchorOutputIndex, reduceDwAxis, "dwh", "dw_hidden", graph,
                               newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  // add db_x / db_h reduce_sum
  vector<int64_t> reduceDbAxis{0, 1};
  isFailure = AddReduceSumNode(gruV2GradNode, gateConcatNode, anchorOutputIndex, reduceDbAxis, "dbx", "db_input", graph,
                               newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDbxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  int anchorDgateIndex = 0;
  if (t_size == 1) {
    anchorDgateIndex = 1;
  }
  ge::NodePtr dbhTransNode = AddNzTransDataNode(gruV2GradNode, hiddenGradNodes["dgate_h"], anchorDgateIndex, "dbh",
                                                graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddNzTransDataNode:check failed, fusion failed."),
                    return FAILED);
  isFailure = AddReduceSumNode(gruV2GradNode, dbhTransNode, 0, reduceDbAxis, "dbh", "db_hidden", graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDbhReduceSumNode:check failed, fusion failed."),
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
      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", gruV2GradNode->GetName().c_str()),
      return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("DynamicGRUV2GradFusionPass", BUILT_IN_GRAPH_PASS, DynamicGRUV2GradFusionPass);
}  // namespace fe
