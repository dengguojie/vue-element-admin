/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * @brief DynamicAUGRUGrad fusion pass(DynamicAUGRUGrad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_augru_grad_align_fusion_pass.h"

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
                                                       {"hidden_new", 9},
                                                       {"seq_mask",    10}};
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
static int64_t INDEX_2 = 2;
vector<FusionPattern*> DynamicAUGRUGradAlignFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicAUGRUGradAlignFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                       "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicAUGRUGradAlignFusionPass::GetNodeInfo(ge::NodePtr dynamicAUGRUGradNode) {
  ge::OpDescPtr dynamicAUGRUGradDesc = dynamicAUGRUGradNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDescH = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  tSize = inputTensorDescH.GetShape().GetDim(0);
  batch = inputTensorDescH.GetShape().GetDim(1);
  nzBatch = (batch + fzDim - 1) / fzDim;
  hiddenDim = inputTensorDescH.GetShape().GetDim(INDEX_2);
  nzHiddenDim = (hiddenDim + fzDim -1) / fzDim;

  ge::GeTensorDesc inputTensorDescX = dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  inputDim = inputTensorDescX.GetShape().GetDim(INDEX_2);
  nzInputDim = (inputDim + fzDim - 1) / fzDim;
  inputHType = inputTensorDescH.GetDataType();
  hasSeqLength = dynamicAUGRUGradNode->GetOpDesc()->MutableInputDesc("seq_length") != nullptr;
  return;
}

void DynamicAUGRUGradAlignFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string &name,
                                                       const vector<int64_t> &dims, const ge::Format &format,
                                                       const vector<int64_t> &originDims,
                                                       const ge::Format &originFormat,
                                                       const ge::DataType &dtype) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  opDesc->AddInputDesc(name, addNodeDesc);
  return;
}

void DynamicAUGRUGradAlignFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                        const vector<int64_t> &dims, const ge::DataType &dtype,
                                                        const ge::Format &format) {
  ge::GeShape originShape(dims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(format);
  opDesc->AddOutputDesc(name, addNodeDesc);
  return;
}

void DynamicAUGRUGradAlignFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                        const vector<int64_t> &dims, const ge::Format &format,
                                                        const vector<int64_t> &originDims,
                                                        const ge::Format &originFormat,
                                                        const ge::DataType &dtype) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginDataType(dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  opDesc->AddOutputDesc(name, addNodeDesc);
  return;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc,
                                                        vector<ge::NodePtr> &newNodes, bool &failStatus) {
  ge::NodePtr node = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(node);
  return node;
}

void DynamicAUGRUGradAlignFusionPass::AddHiddenGradNodeEdge(map<std::string, ge::NodePtr>& inputNodes,
                                                            ge::NodePtr hiddenGradNode, ge::NodePtr matmulGradNode,
                                                            ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                                                            ge::NodePtr genMaskNode,
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
  if (curT < tSize - 1) {
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  } else {
    // fake connect th last cell
    ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  }

  // connect dh_prev to output
  if (curT == tSize) {
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
  if(hasSeqLength){
    ge::GraphUtils::AddEdge(genMaskNode->GetOutDataAnchor(0),
                            hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["seq_mask"]));
  }
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddOneHiddenGradNode(const string& gateOrder, int64_t curT,
                                                                  ge::NodePtr dynamicAUGRUGradNode,
                                                                  ge::ComputeGraph &graph,
                                                                  vector<ge::NodePtr> &newNodes, bool &failStatus) {
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
  if (curT < tSize - 1) {
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
                     {1, batch, hiddenDim}, ge::FORMAT_ND, inputHType);
  }
  hiddenGradDesc->AddInputDesc("update", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["update"]).Clone());
  hiddenGradDesc->AddInputDesc("update_att", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["update_att"]).Clone());
  hiddenGradDesc->AddInputDesc("reset", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["reset"]).Clone());
  hiddenGradDesc->AddInputDesc("new", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["new"]).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());
  // seq_mask has same shapeDesc with hidden_new
  if(hasSeqLength){
    hiddenGradDesc->AddInputDesc("seq_mask", dynamicAUGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());
  }

  vector<int64_t> dgateHNzDim{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dgateHNzDimOri{1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDim{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> singleGateNzDimOri{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dwAttTNzDim{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dwAttTNzDimOri{1, batch, nzHiddenDim * fzDim};
  ge::Format dgateOriFormat = ge::FORMAT_FRACTAL_NZ;
  ge::Format dnxOriFormat = ge::FORMAT_FRACTAL_NZ;

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

void DynamicAUGRUGradAlignFusionPass::AddBatchMatMulForCell(ge::OpDescPtr& lstmBatchMatMulDesc,
                                                            const string &weightName) {
  // add matmul input
  vector<int64_t> LeftDims{nzHiddenDim * (splitSize + 1), nzBatch, fzDim, fzDim};
  vector<int64_t> LeftOriDims{batch, nzHiddenDim * (splitSize + 1) * fzDim};
  vector<int64_t> WeightDims{nzHiddenDim, nzHiddenDim * (splitSize + 1), fzDim, fzDim};
  vector<int64_t> WeightoriDims{hiddenDim, (splitSize + 1) * hiddenDim};
  vector<int64_t> outputDims{1, nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> outputOriDims{1, batch, hiddenDim};
  ge::Format leftOriFormat = ge::FORMAT_FRACTAL_NZ;

  if (weightName == "weight_input") {
    LeftDims = {tSize, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    LeftOriDims = {tSize, batch, nzHiddenDim * (splitSize + 1) * fzDim};
    WeightDims = {nzInputDim, nzHiddenDim * (splitSize + 1), fzDim, fzDim};
    WeightoriDims = {inputDim, 3 * hiddenDim};
    outputDims = {tSize, nzInputDim, nzBatch, fzDim, fzDim};
    outputOriDims = {tSize, batch, inputDim};
    leftOriFormat = ge::FORMAT_ND;
  }

  GeTensorDesc left_tensor_desc = GeTensorDesc(GeShape(LeftDims), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  left_tensor_desc.SetOriginShape(GeShape(LeftOriDims));
  left_tensor_desc.SetOriginFormat(leftOriFormat);
  lstmBatchMatMulDesc->AddInputDesc("dgate", left_tensor_desc);

  GeTensorDesc weight_tensor_desc = GeTensorDesc(GeShape(WeightDims), FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16);
  weight_tensor_desc.SetOriginShape(GeShape(WeightoriDims));
  weight_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddInputDesc("w", weight_tensor_desc);

  // add matmul output
  GeShape outputOriShape(outputOriDims);
  GeShape outputShape(outputDims);
  GeTensorDesc outputTensorDesc = GeTensorDesc(outputShape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  outputTensorDesc.SetOriginShape(outputOriShape);
  outputTensorDesc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddOutputDesc("y", outputTensorDesc);

  // attr
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x1", false);
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x2", true);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "input_size", inputDim);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "hidden_size", hiddenDim);
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddOneHiddenGradMatmulNode(int64_t curT, ge::NodePtr hiddenGradNode,
                                                                        ge::NodePtr dynamicAUGRUGradNode,
                                                                        ge::ComputeGraph &graph,
                                                                        vector<ge::NodePtr> &newNodes,
                                                                        bool &failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (matmulDesc = std::make_shared<ge::OpDesc>(
         dynamicAUGRUGradNode->GetName() + "/AUGRUGrad/Matmul_" + to_string(curT), "BatchMatMulV2")),
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));  // weight
  return matmulNode;
}

vector<vector<ge::NodePtr>> DynamicAUGRUGradAlignFusionPass::AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes,
                                                                          ge::NodePtr dynamicAUGRUGradNode,
                                                                          ge::ComputeGraph &graph,
                                                                          vector<ge::NodePtr> &newNodes,
                                                                          bool &failStatus) {
  ge::OpDescPtr dynamicAUGRUGradDesc = dynamicAUGRUGradNode->GetOpDesc();

  string gateOrder = "zrh";
  ge::AttrUtils::GetStr(dynamicAUGRUGradDesc, "gate_order", gateOrder);

  vector<vector<ge::NodePtr>> result = {};
  vector<ge::NodePtr> hiddenGradNodes = {};
  vector<ge::NodePtr> matmulNodes = {};
  ge::NodePtr lastHiddenGradNode = nullptr;
  ge::NodePtr lastMatmulNode = nullptr;

  ge::NodePtr genMaskNode = nullptr;
  if(hasSeqLength){
    genMaskNode = AddGenMaskNode(dynamicAUGRUGradNode, graph, newNodes, failStatus);
  }

  for (int64_t i = 0; i < tSize; i++) {
    ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, i, dynamicAUGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    ge::NodePtr matmulNode =
        AddOneHiddenGradMatmulNode(i, hiddenGradNode, dynamicAUGRUGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "check failed, fusion failed."), return result);
    // add input edge
    AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, matmulNode, lastHiddenGradNode, lastMatmulNode, genMaskNode,
                          dynamicAUGRUGradNode, i);

    lastHiddenGradNode = hiddenGradNode;
    lastMatmulNode = matmulNode;
    hiddenGradNodes.push_back(hiddenGradNode);
    matmulNodes.push_back(matmulNode);
  }
  // last hiddenGradNode
  ge::NodePtr hiddenGradNode = AddOneHiddenGradNode(gateOrder, tSize, dynamicAUGRUGradNode, graph, newNodes,
                                                    failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return result);
  AddHiddenGradNodeEdge(inputNodes, hiddenGradNode, nullptr, lastHiddenGradNode, lastMatmulNode, genMaskNode,
                        dynamicAUGRUGradNode, tSize);
  hiddenGradNodes.push_back(hiddenGradNode);

  result.push_back(hiddenGradNodes);
  result.push_back(matmulNodes);
  return result;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddTConcatNode(const string& nodeName, const string& inputName,
                                                            vector<int64_t> fzDims, ge::NodePtr dynamicAUGRUGradNode,
                                                            vector<ge::NodePtr> &srcNodes, ge::ComputeGraph &graph,
                                                            vector<ge::NodePtr> &newNodes, bool &failStatus) {
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
  for (int64_t i = 0; i < tSize; i++) {
    concatDesc->AddInputDesc("input_" + to_string(i), inputDesc);
  }

  // output concat, shape:{t,batch_size,hidden_size}
  GeTensorDesc outputDesc = srcNodes[0]->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX[inputName]).Clone();
  vector<int64_t> outDim = outputDesc.GetShape().GetDims();
  vector<int64_t> outDimOri = outputDesc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(outDim.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                   "AddTConcatNode:check failed, fusion failed."),
                    return nullptr);
  outDim[0] = tSize;
  outDimOri[0] = tSize;
  outputDesc.SetShape(GeShape(outDim));
  outputDesc.SetOriginShape(GeShape(outDimOri));
  concatDesc->AddOutputDesc("concat_" + inputName, outputDesc);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", tSize);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."), return nullptr);

  // Edge
  for (int64_t i = 0; i < tSize; i++) {
    ge::GraphUtils::AddEdge(srcNodes[i]->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX[inputName]),
                            concatNode->GetInDataAnchor(tSize - 1 - i));  // Init_h
  }
  return concatNode;
}

map<std::string, ge::NodePtr> DynamicAUGRUGradAlignFusionPass::AddGRUHiddenGradNode(ge::NodePtr dynamicAUGRUGradNode,
                                                                                    ge::ComputeGraph &graph,
                                                                                    vector<ge::NodePtr> &newNodes,
                                                                                    bool &failStatus) {
  map<std::string, ge::NodePtr> inputNodes;
  map<std::string, ge::NodePtr> result;
  vector<vector<ge::NodePtr>> result_node;
  if (tSize > 1) {
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
    dntXConcatTNode = AddTConcatNode("/AUGRUGrad/ConcatDntX", "dnt_x", fzDims, dynamicAUGRUGradNode,
                                     result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "AddDntXConcatTNode:check failed, fusion failed."),
                      return result);

    // add dgate_h concat node
    fzDims = {1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
    ge::NodePtr dgateHConcatTNode = nullptr;
    dgateHConcatTNode = AddTConcatNode("/AUGRUGrad/ConcatDgateH", "dgate_h", fzDims, dynamicAUGRUGradNode,
                                       result_node[0], graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                 "AddDgateHConcatTNode:check failed, fusion failed."),
                      return result);

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

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddGenMaskNode(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph &graph,
                                                            vector<ge::NodePtr> &newNodes, bool &failStatus) {
  ge::OpDescPtr genMaskDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (genMaskDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUweightGrad/GenMaskNode",
                                                  "RnnGenMask")),
      genMaskDesc = nullptr;
          failStatus = true;
          return nullptr);
  // input
  vector<int64_t> inputDims = {batch};
  AddInputNodeDesc(genMaskDesc, "seq_length", inputDims, ge::FORMAT_ND, inputDims, ge::FORMAT_ND,
                   ge::DT_INT32);

  // output
  vector<int64_t> dstDims = {tSize, batch, hiddenDim};
  AddOutputNodeDesc(genMaskDesc, "seq_mask", dstDims, ge::FORMAT_ND, dstDims, ge::FORMAT_ND, ge::DT_FLOAT16);

  // attr
  ge::AttrUtils::SetInt(genMaskDesc, "num_step", tSize);
  ge::AttrUtils::SetInt(genMaskDesc, "hidden_size", hiddenDim);

  // create node
  ge::NodePtr genMaskNode = AddNewNode(graph, genMaskDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["seq_length"])->GetPeerOutAnchor(),
                          genMaskNode->GetInDataAnchor(0));
  return genMaskNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddHTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph &graph,
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
  vector<int64_t> dstDims = {tSize, nzHiddenDim, nzBatch, 16, 16};
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

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddHSplitNode(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr> &newNodes, bool &failStatus) {
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
  splitDesc->AddInputDesc("input_h", inputTensorDescH);

  vector<int64_t> size_splits = {(tSize - 1) * batch, batch};
  // add output1 split_t_1, shape:{t-1,batch_size,hidden_size}
  vector<int64_t> output1NzDim{tSize - 1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_t_1", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_t_1

  // add output2 split_1, shape:{1,batch_size,hidden_size}
  vector<int64_t> output2NzDim{1, nzHiddenDim, nzBatch, 16, 16};
  AddOutputNodeDesc(splitDesc, "split_1", output2NzDim, ge::FORMAT_FRACTAL_NZ, output2NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_1
  // attr
  size_splits = {tSize - 1, 1};
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

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDwhTransData(ge::NodePtr dynamicAUGRUGradNode, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr> &newNodes, bool &failStatus) {
  ge::OpDescPtr transDataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transDataDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUweightGrad/Dwh/TransData",
                                                    "TransData")),
      transDataDesc = nullptr;
      failStatus = true;
      return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicAUGRUGradNode->GetOpDesc()->GetInputDesc(
      INPUT_INDEX["init_h"]).Clone();
  inputTensorDescInitH.SetShape(GeShape({1, batch, hiddenDim}));
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

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddHConcatNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr splitNode,
                                                            ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                                            bool &failStatus) {
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
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output concat_h, shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDim{tSize, batch, hiddenDim};
  vector<int64_t> outputNzDim{tSize, nzHiddenDim, nzBatch, 16, 16};
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

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr hConcatNode,
                                                              ge::NodePtr gruHiddenGradNode, ge::ComputeGraph &graph,
                                                              vector<ge::NodePtr> &newNodes, bool &failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (matmulDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul",
                                               "BatchMatMul")),
    matmulDesc = nullptr;
    failStatus = true;
    return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  // gruHiddenGradNode is dgateConcatNode
  ge::GeTensorDesc inputTensorDescDgate;
  if (tSize == 1) {
    inputTensorDescDgate =
        gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();  // dgate_h
  } else {
    inputTensorDescDgate = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();  // dgate_h
  }
  inputTensorDescH.SetDataType(ge::DT_FLOAT16);
  inputTensorDescH.SetOriginShape(GeShape({tSize, batch, nzHiddenDim * fzDim}));
  inputTensorDescDgate.SetOriginShape(GeShape({tSize, batch, (splitSize + 1) * nzHiddenDim * fzDim}));
  inputTensorDescDgate.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);

  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector <int64_t> outputDim{tSize, nzHiddenDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{tSize, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
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
  if (tSize == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            matmulNode->GetInDataAnchor(1));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_h
  }
  return matmulNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicAUGRUGradNode,
                                                              ge::NodePtr gruHiddenGradNode,
                                                              ge::ComputeGraph &graph,
                                                              vector<ge::NodePtr> &newNodes,
                                                              bool &failStatus) {
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
  if (tSize == 1) {
    inputTensorDescDgate =
        gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();  // dgate_h
  } else {
    inputTensorDescDgate = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }

  inputTensorDescH.SetShape(GeShape({1, nzHiddenDim, nzBatch, 16, 16}));
  inputTensorDescH.SetOriginShape(GeShape({1, batch, hiddenDim}));
  inputTensorDescH.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescH.SetDataType(ge::DT_FLOAT16);

  inputTensorDescDgate.SetShape(GeShape({1, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim}));
  inputTensorDescDgate.SetOriginShape(GeShape({1, batch, (splitSize + 1) * nzHiddenDim * fzDim}));
  inputTensorDescDgate.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("input_h", inputTensorDescH);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  // add output dwt_h shape:{t, hidden_size, 3 * hide_size}
  vector<int64_t> outputDim{tSize, nzHiddenDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{tSize, (splitSize + 1) * nzHiddenDim, nzHiddenDim, fzDim, fzDim};
  if (tSize == 1) {
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));  // Init_h
  if (tSize == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            matmulNode->GetInDataAnchor(1));  // dgate_h
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_h
  }
  return matmulNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDgateHSplitNode(ge::NodePtr dynamicAUGRUGradNode,
                                                                ge::NodePtr gruHiddenGradNode, ge::ComputeGraph &graph,
                                                                vector<ge::NodePtr> &newNodes, bool &failStatus) {
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
  if (tSize == 1) {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  } else {
    inputDgateH = gruHiddenGradNode->GetOpDesc()->GetOutputDesc(0).Clone();
  }
  splitDesc->AddInputDesc("input_dgate_h", inputDgateH);

  // add output1 dgate_ir, shape:{t, batch, 2 * hidden_size}
  vector<int64_t> output1Dim{tSize, batch, splitSize * hiddenDim};
  vector<int64_t> output1NzDim{tSize, splitSize * nzHiddenDim, nzBatch, fzDim, fzDim};
  AddOutputNodeDesc(splitDesc, "split_ir", output1NzDim, ge::FORMAT_FRACTAL_NZ, output1NzDim, ge::FORMAT_FRACTAL_NZ,
                    inputHType);  // split_didr

  // add output2 split_1, shape:{t, batch, hidden_size}
  vector<int64_t> output2NzDim{tSize, nzHiddenDim, nzBatch, fzDim, fzDim};
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
  if (tSize == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                            splitNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), splitNode->GetInDataAnchor(0));
  }
  return splitNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDgateXConcatNode(ge::NodePtr dynamicAUGRUGradNode,
                                                                 ge::NodePtr dgateHSplitNode,
                                                                 ge::NodePtr gruHiddenGradNode, ge::ComputeGraph &graph,
                                                                 vector<ge::NodePtr> &newNodes, bool &failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD",
                                                 "ConcatD")),
      concatDesc = nullptr;
      failStatus = true;
      return nullptr);

  // input
  vector<int64_t> dirtNzDesc = {tSize, splitSize * nzHiddenDim, nzBatch, fzDim, fzDim};
  vector<int64_t> dnxNzDesc = {tSize, nzHiddenDim, nzBatch, fzDim, fzDim};
  AddInputNodeDesc(concatDesc, "input_dirt", dirtNzDesc, ge::FORMAT_FRACTAL_NZ, dirtNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);
  AddInputNodeDesc(concatDesc, "input_dnt_x", dnxNzDesc, ge::FORMAT_FRACTAL_NZ, dnxNzDesc, ge::FORMAT_FRACTAL_NZ,
                   inputHType);

  // output shape:{t,batch,3*hidden_size}
  vector<int64_t> outputDim{tSize, batch, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{tSize, (splitSize + 1) * nzHiddenDim, nzBatch, fzDim, fzDim};
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
  if (tSize == 1) {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dnt_x"]),
                            concatNode->GetInDataAnchor(1));
  } else {
    ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  }
  return concatNode;
}

Status DynamicAUGRUGradAlignFusionPass::AddDxtMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                         ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulOpDesc = std::make_shared<ge::OpDesc>(dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmulV2",
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_input"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(1));

  // output Edge
  for (InDataAnchorPtr inAnchorPtr : dynamicAUGRUGradNode->GetOutDataAnchor(
      OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
    // dxt
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return failStatus;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDwxMatmulNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                              ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                                              bool &failStatus) {
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
  xtDesc.SetShape(GeShape({tSize, nzInputDim, nzBatch, fzDim, fzDim}));
  dgateXDesc.SetDataType(ge::DT_FLOAT16);
  matmulDesc->AddInputDesc("xt", xtDesc);
  matmulDesc->AddInputDesc("dgate_x", dgateXDesc);

  // add output dwx, shape:{t, inputDim, 3 * hiddenDim}
  vector<int64_t> outputDim{tSize, nzInputDim * fzDim, (splitSize + 1) * nzHiddenDim * fzDim};
  vector<int64_t> outputNzDim{tSize, (splitSize + 1) * nzHiddenDim, nzInputDim, fzDim, fzDim};
  if (tSize == 1) {
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
  ge::GraphUtils::AddEdge(dynamicAUGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));                                         // xt
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate_x
  return matmulNode;
}

ge::OpDescPtr DynamicAUGRUGradAlignFusionPass::SetDescForTransdata(ge::OpDescPtr &transdataDesc,
                                                                   const string& srcFormat,
                                                                   const string& weightName) {
  // input for transdata
  int64_t dim0 = inputDim;
  int64_t nzdim0 = nzInputDim;
  if (weightName == "weight_hidden") {
    dim0 = hiddenDim;
    nzdim0 = nzHiddenDim;
  }
  vector<int64_t> transDims{nzdim0, nzHiddenDim * 3, 16, 16};
  vector<int64_t> transOriDims{dim0, hiddenDim * 3};
  ge::Format transFormat = ge::FORMAT_FRACTAL_ZN_RNN;
  if (srcFormat == "ND_RNN_BIAS") {
    transDims = {nzHiddenDim * 3 * 16};
    transOriDims = {hiddenDim * 3};
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
  AttrUtils::SetInt(transdataDesc, "input_size", inputDim);
  AttrUtils::SetInt(transdataDesc, "hidden_size", hiddenDim);
  return transdataDesc;
}

ge::OpDescPtr DynamicAUGRUGradAlignFusionPass::SetDescForTranspose(ge::OpDescPtr &transposeDesc,
                                                                   const string &weightName) {
  // input for transdata
  int64_t dim0 = inputDim;
  int64_t nzdim0 = nzInputDim;
  if (weightName == "weight_hidden") {
    dim0 = hiddenDim;
    nzdim0 = nzHiddenDim;
  }
  vector<int64_t> transInputDims{nzHiddenDim * 3, nzdim0, 16, 16};
  vector<int64_t> transInputOriDims{nzdim0 * 16, nzHiddenDim * 16 * 3};
  vector<int64_t> transOutputDims{nzdim0, nzHiddenDim * 3, 16, 16};
  vector<int64_t> transOutputOriDims{dim0, hiddenDim * 3};

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
  ge::AttrUtils::SetInt(transposeDesc, "input_size", inputDim);
  ge::AttrUtils::SetInt(transposeDesc, "hidden_size", hiddenDim);

  return transposeDesc;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDbReduceSumTransNode(
    ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode, int anchorIndex, const vector<int64_t>& axis,
    const string& nodeName, const string& indexName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
    const vector<int64_t>& transDims, bool& failStatus) {
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

  ge::Format transFormat = ge::FORMAT_ND_RNN_BIAS;
  // output
  ge::GeTensorDesc reduceOutputTensorDesc = ge::GeTensorDesc(GeShape(transDims), transFormat, ge::DT_FLOAT16);
  vector<int64_t> outputOriDims{hiddenDim * 3};
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

  // create trans_data_rnn node
  ge::OpDescPtr transdataDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdataDesc = std::make_shared<ge::OpDesc>(
      dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransDataRNN", "TransDataRNN")),
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
      dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddDwReduceSumTransNode(
    ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode, int anchorIndex, const vector<int64_t>& axis,
    const string& nodeName, const string& indexName, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
    const vector<int64_t>& transDims, const string& weightName, bool& failStatus) {
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
      dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransPose", "TransposeD")),
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
      dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddTransposeNode(
    ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dwMatmulNode, const string &nodeName, const string &weightName,
    const string &outputName, ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes) {
  // insert transpose
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
      dynamicAUGRUGradNode->GetName() + "GRUWeightGrad/" + nodeName + "/TransPose", "TransposeD")),
  return nullptr);
  transposeDesc = SetDescForTranspose(transposeDesc, weightName);
  bool failStatus = false;
  ge::NodePtr transposeNode = this->AddNewNode(graph, transposeDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
  return nullptr);
  ge::GraphUtils::AddEdge(dwMatmulNode->GetOutDataAnchor(0), transposeNode->GetInDataAnchor(0));
  for (InDataAnchorPtr inAnchorPtr :
      dynamicAUGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[outputName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return transposeNode;
}

ge::NodePtr DynamicAUGRUGradAlignFusionPass::AddTReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr inputNode,
                                                               int anchorIndex, const vector<int64_t> &axis,
                                                               const string &nodeName,
                                                               ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                                               bool &failStatus) {
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
Status DynamicAUGRUGradAlignFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicAUGRUGradNode, ge::NodePtr dwxMatmulNode,
                                                           ge::NodePtr dwhMatmulNode, ge::ComputeGraph &graph,
                                                           vector<ge::NodePtr> &newNodes) {
  // add dw_x / dw_h reduce_sum
  if (tSize == 1) {
    // no need reduce_sum
    AddTransposeNode(dynamicAUGRUGradNode, dwxMatmulNode, "dwx", "weight_input", "dw_input", graph, newNodes);
    AddTransposeNode(dynamicAUGRUGradNode, dwhMatmulNode, "dwh", "weight_hidden", "dw_hidden", graph, newNodes);
    return SUCCESS;
  }
  int anchorOutputIndex = 0;
  vector<int64_t> reduceDwAxis{0};
  bool isFailure = false;
  AddDwReduceSumTransNode(dynamicAUGRUGradNode, dwxMatmulNode, anchorOutputIndex, reduceDwAxis, "dwx", "dw_input",
                          graph, newNodes, {3 * nzHiddenDim, nzInputDim, fzDim, fzDim}, "weight_input", isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddDwReduceSumTransNode(dynamicAUGRUGradNode, dwhMatmulNode, anchorOutputIndex, reduceDwAxis, "dwh", "dw_hidden",
                          graph, newNodes, {3 * nzHiddenDim, nzHiddenDim, fzDim, fzDim}, "weight_hidden", isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

Status DynamicAUGRUGradAlignFusionPass::AddDbReduceSumNode(ge::NodePtr augruGradNode, ge::NodePtr dbxNode,
                                                           ge::NodePtr dbhNode, ge::ComputeGraph &graph,
                                                           vector<ge::NodePtr> &newNodes) {
  // add db_x / db_h reduce_sum
  int anchorOutputIndex = 0;
  bool isFailure = false;
  vector<int64_t> reduceDbAxis{2, 3};
  if (tSize == 1) {
    // NZ {1, 3 * nzHiddenDim, nzBatch, 16, 16}
    // ND {1, batch, 3*hidden}
    vector<int64_t> reduceDbxAxis{1};
    AddDbReduceSumTransNode(augruGradNode, dbxNode, anchorOutputIndex, reduceDbxAxis, "dbx", "db_input",
                            graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbxReduceSumNode:check failed, fusion failed."),
                      return FAILED);

    anchorOutputIndex = 1;
    AddDbReduceSumTransNode(augruGradNode, dbhNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden",
                            graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
    FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                                "AddDbhReduceSumNode:check failed, fusion failed."),
                      return FAILED);
    return SUCCESS;
  }
  // {tSize, 3 * nzHiddenDim, nzBatch, 16, 16}
  vector<int64_t> reduceDbTAxis{0};
  // dbx
  ge::NodePtr dbxTReduceSumNode = AddTReduceSumNode(augruGradNode, dbxNode, anchorOutputIndex, reduceDbTAxis,
                                                    "dbx_t", graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbxTReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddDbReduceSumTransNode(augruGradNode, dbxTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbx", "db_input",
                          graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  // dbh
  ge::NodePtr dbhTReduceSumNode = AddTReduceSumNode(augruGradNode, dbhNode, anchorOutputIndex, reduceDbTAxis,
                                                    "dbh_t", graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbhTReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddDbReduceSumTransNode(augruGradNode, dbhTReduceSumNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden",
                          graph, newNodes, {3 * nzHiddenDim * fzDim}, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDbhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

Status DynamicAUGRUGradAlignFusionPass::AddDwAttReduceSumNode(ge::NodePtr augruGradNode, ge::NodePtr dwAttNode,
                                                              ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(augruGradNode->GetName() + "GRUWeightGrad/dwAtt/ReduceSumD",
                                                    "ReduceSumD")), reduceSumDesc = nullptr; return FAILED);

  // input
  vector<int64_t> dwAttInshape = {tSize, batch, nzHiddenDim * fzDim};
  AddInputNodeDesc(reduceSumDesc, "x", dwAttInshape, ge::FORMAT_ND, dwAttInshape, ge::FORMAT_ND,
                   inputHType);

  // output
  vector<int64_t> dwAttOutshape = {tSize, batch};
  AddOutputNodeDesc(reduceSumDesc, "y", dwAttOutshape, ge::FORMAT_ND, dwAttOutshape,
                    ge::FORMAT_ND, inputHType);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {2});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  bool failStatus = false;
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                               "check failed, fusion failed."),
                    return FAILED);

  // Edge
  int anchorOutputIndex = 0;
  if (tSize == 1) {
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

Status DynamicAUGRUGradAlignFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicAUGRUGradAlignFusionPass fusion begin.");
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
  hiddenDim = inputTensorDescH.GetShape().GetDim(splitSize);
  inputDim = inputTensorDescX.GetShape().GetDim(splitSize);

  tSize = inputTensorDescH.GetShape().GetDim(0);
  if (hiddenDim % fzDim == 0 && inputDim % fzDim == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is 16 align, will not changed");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(batch) ||
      PatternFusionUtil::IsUnknownShape(hiddenDim) || PatternFusionUtil::IsUnknownShape(tSize) ||
      PatternFusionUtil::IsUnknownShape(inputDim)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "DynamicAUGRUGradAlignFusionPass cannot be applied for unknown shape.");
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
  if (tSize != 1) {
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
  dgateHSplitNode = AddDgateHSplitNode(augruGradNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // concat [dit, drt] with [dnt_x] to dgate_x
  ge::NodePtr gateConcatNode = nullptr;
  gateConcatNode = AddDgateXConcatNode(augruGradNode, dgateHSplitNode,
                                       hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDgateXConcatNode:check failed, fusion failed."),
                    return FAILED);

  // add dxt matmul(dgate_x, w_x.T)
  isFailure = AddDxtMatmulNode(augruGradNode, gateConcatNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDxtMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x matmul(x.T, dgate_x)
  ge::NodePtr dwxMatmulNode = nullptr;
  dwxMatmulNode = AddDwxMatmulNode(augruGradNode, gateConcatNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwxMatmulNode:check failed, fusion failed."),
                    return FAILED);
  isFailure = AddDwReduceSumNode(augruGradNode, dwxMatmulNode, dwhMatmulNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                              "AddDwReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddDbReduceSumNode(augruGradNode, gateConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes);
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

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicAUGRUGradAlignFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("DynamicAUGRUGradAlignFusionPass", BUILT_IN_GRAPH_PASS, DynamicAUGRUGradAlignFusionPass);
}  // namespace fe
