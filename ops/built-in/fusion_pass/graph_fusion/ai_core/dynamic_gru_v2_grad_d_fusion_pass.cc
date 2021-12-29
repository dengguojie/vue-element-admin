/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2Grad fusion pass(DynamicGRUV2Grad --> GRUHiddenGrad & GRUWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_gru_v2_grad_d_fusion_pass.h"

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
#include "graph/utils/type_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "DynamicGRUV2Grad";
static const std::string PATTERN_FUSEDNODE = "DynamicGRUV2Grad";
static const int INDEX_2 = 2;
static const int INDEX_3 = 3;
static const int INDEX_4 = 4;
static const int INDEX_5 = 5;
static const int INDEX_6 = 6;
static const int INDEX_7 = 7;
static const int INDEX_8 = 8;
static const int INDEX_9 = 9;
static const int INDEX_10 = 10;
static const int INDEX_11 = 11;
static const int INDEX_12 = 12;
static const int INDEX_13 = 13;
static const int INDEX_14 = 14;
static const int DIM_NUM_2 = 2;
static const int CONCAT_NUM = 2;
static const int HIDDEN_NUM = 3;
static const int ALIGN_16 = 16;
static const int BATCH_32 = 32;

static map<std::string, int> INPUT_INDEX = {
    {"x", 0},           {"weight_input", 1}, {"weight_hidden", 2}, {"y", 3},     {"init_h", 4}, {"h", 5},
    {"dy", 6},          {"dh", 7},           {"update", 8},        {"reset", 9}, {"new", 10},   {"hidden_new", 11},
    {"seq_length", 12}, {"mask", 13}};

static map<std::string, int> HIDDENGRAD_INPUT_INDEX = {{"dh_pre_t", 0}, {"h", 1},      {"dy", 2},  {"dh", 3},
                                                       {"update", 4},   {"reset", 5},  {"new", 6}, {"hidden_new", 7},
                                                       {"init_h", 8},   {"t_state", 9}};
static map<std::string, int> OUTPUT_INDEX = {{"dw_input", 0},  {"dw_hidden", 1}, {"db_input", 2},
                                             {"db_hidden", 3}, {"dx", 4},        {"dh_prev", 5}};
static map<std::string, int> HIDDENGRAD_OUTPUT_INDEX = {{"dh_prev", 0}, {"dgate_h", 1}, {"dnt_x", 2}};

vector<FusionPattern*> DynamicGRUV2GradDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicGRUV2GradDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicGRUV2GradDFusionPass::GetNodeInfo(ge::NodePtr dynamicGRUGradNode) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);

  batch = inputTensorDescH.GetShape().GetDim(1);
  if (batch == -1) {
    nzBatch = -1;
    batch_start = 1;
    batch_end = BATCH_32;
  } else {
    nzBatch = (batch + ALIGN_16 - 1) / ALIGN_16;
    batch_start = nzBatch;
    batch_end = nzBatch;
  }
  hidden_dim = inputTensorDescH.GetShape().GetDim(DIM_NUM_2);
  nzHiddenDim = (hidden_dim + ALIGN_16 - 1) / ALIGN_16;
  ge::GeTensorDesc inputTensorDescX = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);
  input_dim = inputTensorDescX.GetShape().GetDim(DIM_NUM_2);
  nzInputDim = (input_dim + ALIGN_16 - 1) / ALIGN_16;
  inputHType = inputTensorDescH.GetDataType();
  return;
}

void DynamicGRUV2GradDFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

void DynamicGRUV2GradDFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                   const vector<int64_t>& dims, const ge::Format& format,
                                                   const vector<int64_t>& originDims, const ge::Format& originFormat,
                                                   const ge::DataType& dtype,
                                                   std::vector<std::pair<int64_t, int64_t>> x_range) {
  ge::GeShape originShape(originDims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(originFormat);
  addNodeDesc.SetOriginShapeRange(x_range);
  addNodeDesc.SetShapeRange(x_range);
  opDesc->AddInputDesc(name, addNodeDesc);
  return;
}

void DynamicGRUV2GradDFusionPass::AddInputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                   const vector<int64_t>& dims, const ge::Format& format,
                                                   const ge::DataType& dtype) {
  ge::GeShape originShape(dims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(format);
  opDesc->AddInputDesc(name, addNodeDesc);
  return;
}

void DynamicGRUV2GradDFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
                                                    const vector<int64_t>& dims, const ge::DataType& dtype,
                                                    const ge::Format& format) {
  ge::GeShape originShape(dims);
  ge::GeShape curShape(dims);
  ge::GeTensorDesc addNodeDesc = ge::GeTensorDesc(curShape, format, dtype);
  addNodeDesc.SetOriginShape(originShape);
  addNodeDesc.SetOriginFormat(format);
  addNodeDesc.SetOriginDataType(dtype);
  opDesc->AddOutputDesc(name, addNodeDesc);
  return;
}

void DynamicGRUV2GradDFusionPass::AddOutputNodeDesc(ge::OpDescPtr opDesc, const std::string& name,
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

ge::NodePtr DynamicGRUV2GradDFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& opDesc,
                                                    vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::NodePtr node = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", node->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(node);
  return node;
}

void DynamicGRUV2GradDFusionPass::AddT0GradNodeEdge(map<std::string, ge::NodePtr>& inputNodes,
                                                    ge::NodePtr hiddenGradNode, ge::NodePtr matmulGradNode,
                                                    ge::NodePtr lastHiddenGradNode, ge::NodePtr lastMatmulNode,
                                                    ge::NodePtr dynamicGRUGradNode) {
  // fake connect dh_pre_t
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh_pre_t"]));
  // connect dh
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh"]));
  // connect h
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          hiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["init_h"]));
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
  // matmul
  ge::GraphUtils::AddEdge(hiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]),
                          matmulGradNode->GetInDataAnchor(0));  // dgate_h
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          matmulGradNode->GetInDataAnchor(1));  // weight
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildT0Cell(const string& gateOrder, ge::GeTensorDesc tStateDesc,
                                                     ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::OpDescPtr hiddenGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((hiddenGradDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "/GRUV2Grad/GRUV2HiddenGradCell_" + std::to_string(0),
                               "DynamicGRUCellGrad")),
                          hiddenGradDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // set attr of gate order
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);

  // set input desc
  ge::GeTensorDesc dhPrevDesc = dynamicGRUGradDesc->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).Clone();
  hiddenGradDesc->AddInputDesc("dh_pre_t", dhPrevDesc);
  ge::GeTensorDesc hDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]).Clone();
  hiddenGradDesc->AddInputDesc("h", hDesc);
  ge::GeTensorDesc dyDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dy"]).Clone();
  hiddenGradDesc->AddInputDesc("dy", dyDesc);
  ge::GeTensorDesc dhDesc = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dh"]).Clone();
  hiddenGradDesc->AddInputDesc("dh", dhDesc);
  hiddenGradDesc->AddInputDesc("update", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["update"]).Clone());
  hiddenGradDesc->AddInputDesc("reset", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["reset"]).Clone());
  hiddenGradDesc->AddInputDesc("new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["new"]).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());
  hiddenGradDesc->AddInputDesc("init_h", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["init_h"]).Clone());
  hiddenGradDesc->AddInputDesc("t_state", tStateDesc);

  // set output desc

  vector<int64_t> dgateHDims{1, batch, HIDDEN_NUM * hidden_dim};
  vector<int64_t> singleGateDims{1, batch, hidden_dim};
  ge::Format dFormat = ge::FORMAT_ND;

  hiddenGradDesc->AddOutputDesc("dh_prev", dhPrevDesc);
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHDims, inputHType, dFormat);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateDims, inputHType, dFormat);

  // create gru_hidden_grad node
  ge::NodePtr hiddenGradNode = this->AddNewNode(graph, hiddenGradDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."),
                    return nullptr);

  return hiddenGradNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildShape(ge::GeTensorDesc xDesc, ge::NodePtr dynamicGRUGradNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                    bool& failStatus) {
  ge::OpDescPtr shapeDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "/GRUV2Grad/shape", "Shape");
  shapeDesc->AddInputDesc("x", xDesc);
  vector<int64_t> inputDims = xDesc.GetShape().GetDims();
  AddOutputNodeDesc(shapeDesc, "y", {static_cast<int64_t>(inputDims.size())},
                    ge::FORMAT_ND, {static_cast<int64_t>(inputDims.size())}, ge::FORMAT_ND, ge::DT_INT32);
  ge::NodePtr shapeNode = this->AddNewNode(graph, shapeDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."),
                    return nullptr);

  return shapeNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddOneHiddenGradNode(const string& gateOrder, ge::NodePtr tSizeConst,
                                                              ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  ge::OpDescPtr tSizeConstDesc = tSizeConst->GetOpDesc();

  ge::OpDescPtr hiddenGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (hiddenGradDesc = std::make_shared<ge::OpDesc>(
           dynamicGRUGradNode->GetName() + "/GRUV2Grad/GRUV2HiddenGradCell_LastT", "DynamicGRUCellGrad")),
      hiddenGradDesc = nullptr;
      failStatus = true;
      return nullptr);

  // set attr of gate order
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);

  // set input desc
  ge::GeTensorDesc dhPrevDesc = dynamicGRUGradDesc->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).Clone();
  hiddenGradDesc->AddInputDesc("dh_pre_t", dhPrevDesc);
  hiddenGradDesc->AddInputDesc("h", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]).Clone());
  hiddenGradDesc->AddInputDesc("dy", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dy"]).Clone());
  hiddenGradDesc->AddInputDesc("dh", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dh"]).Clone());
  hiddenGradDesc->AddInputDesc("update", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["update"]).Clone());
  hiddenGradDesc->AddInputDesc("reset", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["reset"]).Clone());
  hiddenGradDesc->AddInputDesc("new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["new"]).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone());
  hiddenGradDesc->AddInputDesc("init_h", dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["init_h"]).Clone());
  hiddenGradDesc->AddInputDesc("t_state", tSizeConstDesc->GetOutputDesc(0).Clone());

  // set output desc
  vector<int64_t> dgateHDims{1, batch, HIDDEN_NUM * hidden_dim};
  vector<int64_t> singleGateDims{1, batch, hidden_dim};
  ge::Format dFormat = ge::FORMAT_ND;

  hiddenGradDesc->AddOutputDesc("dh_prev", dhPrevDesc);
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHDims, inputHType, dFormat);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateDims, inputHType, dFormat);

  // create gru_hidden_grad node
  ge::NodePtr hiddenGradNode = this->AddNewNode(graph, hiddenGradDesc, newNodes, failStatus);
  return hiddenGradNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddT0MatmulNode(ge::NodePtr hiddenGradNode, ge::NodePtr dynamicGRUGradNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;

  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "/GRUV2Grad/Matmul_T0", "BatchMatMul")),
                          matmulDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // set x1 shape range
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  x1_range.insert(x1_range.begin(), std::make_pair(1, 1));

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim, hidden_dim));

  // input
  ge::GeTensorDesc inputDesc = hiddenGradNode->GetOpDesc()->GetOutputDesc(HIDDENGRAD_OUTPUT_INDEX["dgate_h"]).Clone();
  inputDesc.SetShapeRange(x1_range);
  // weight
  ge::GeTensorDesc weightDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone();
  weightDesc.SetShapeRange(x2_range);

  matmulDesc->AddInputDesc("x1", inputDesc);
  matmulDesc->AddInputDesc("x2", weightDesc);

  vector<int64_t> outputDim{1, batch, hidden_dim};
  AddOutputNodeDesc(matmulDesc, "y", outputDim, inputHType, ge::FORMAT_ND);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", true);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  return matmulNode;
}

vector<ge::NodePtr> DynamicGRUV2GradDFusionPass::AddTLoopNode(map<std::string, ge::NodePtr>& inputNodes,
                                                              ge::NodePtr dynamicGRUGradNode, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();

  string gateOrder = "zrh";
  ge::AttrUtils::GetStr(dynamicGRUGradDesc, "gate_order", gateOrder);

  vector<ge::NodePtr> result = {};
  vector<ge::NodePtr> hiddenGradNodes = {};
  vector<ge::NodePtr> matmulNodes = {};
  ge::NodePtr lastHiddenGradNode = nullptr;
  ge::NodePtr lastMatmulNode = nullptr;

  ge::OpDescPtr t0Const = CreateConstDesc("t0Const", 0, "int32");
  ge::NodePtr t0ConstNode = graph.AddNode(t0Const);
  newNodes.push_back(t0ConstNode);

  ge::NodePtr t0HiddenGradNode = BuildT0Cell(gateOrder, t0ConstNode->GetOpDesc()->GetOutputDesc(0).Clone(),
                                             dynamicGRUGradNode, graph, newNodes, failStatus);

  // build list const
  vector<int64_t> listValue = {0, };
  ge::OpDescPtr uniqueInput = CreateListConstDesc("uniqueInput", listValue);
  ge::NodePtr listConst = AddNewNode(graph, uniqueInput, newNodes, failStatus);

  // build unique
  ge::NodePtr unique = BuildUnique("t0dynamic", newNodes, failStatus, graph);
  // build hGather
  ge::GeTensorDesc tanInputDesc = t0HiddenGradNode->GetOpDesc()->GetOutputDesc(1).Clone();
  vector<int64_t> out_unique_shape = tanInputDesc.GetShape().GetDims();
  out_unique_shape[0] = -1;
  ge::GeTensorDesc gatherHoutDesc = ge::GeTensorDesc(GeShape(out_unique_shape), ge::FORMAT_ND, inputHType);
  gatherHoutDesc.SetOriginShape(GeShape(out_unique_shape));
  ge::NodePtr gatherH = BuildGather("t0HGather", tanInputDesc, unique->GetOpDesc()->GetOutputDesc(0).Clone(),
                                    gatherHoutDesc, newNodes, failStatus, graph);
  ge::GraphUtils::AddEdge(listConst->GetOutDataAnchor(0), unique->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(t0HiddenGradNode->GetOutDataAnchor(1), gatherH->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(unique->GetOutDataAnchor(0), gatherH->GetInDataAnchor(1));

  // build t0XGather
  ge::GeTensorDesc tanxInputDesc = t0HiddenGradNode->GetOpDesc()->GetOutputDesc(INDEX_2).Clone();
  ge::GeTensorDesc gatherXOutDesc = t0HiddenGradNode->GetOpDesc()->GetOutputDesc(INDEX_2).Clone();
  vector<int64_t> out_unique_x_shape = gatherXOutDesc.GetShape().GetDims();
  out_unique_x_shape[0] = -1;
  gatherXOutDesc.SetShape(GeShape(out_unique_x_shape));
  gatherXOutDesc.SetFormat(ge::FORMAT_ND);
  gatherXOutDesc.SetOriginShape(GeShape(out_unique_x_shape));
  gatherXOutDesc.SetOriginFormat(tanxInputDesc.GetFormat());
  ge::NodePtr gatherX = BuildGather("t0XGather", tanxInputDesc, unique->GetOpDesc()->GetOutputDesc(0).Clone(),
                                    gatherXOutDesc, newNodes, failStatus, graph);

  ge::GraphUtils::AddEdge(t0HiddenGradNode->GetOutDataAnchor(INDEX_2), gatherX->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(unique->GetOutDataAnchor(0), gatherX->GetInDataAnchor(1));

  // add t0 matmul
  ge::NodePtr t0MatMulNode = AddT0MatmulNode(t0HiddenGradNode, dynamicGRUGradNode, graph, newNodes, failStatus);
  AddT0GradNodeEdge(inputNodes, t0HiddenGradNode, t0MatMulNode, lastHiddenGradNode, lastMatmulNode, dynamicGRUGradNode);
  ge::GraphUtils::AddEdge(t0ConstNode->GetOutDataAnchor(0),
                          t0HiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["t_state"]));

  // build currT const
  ge::OpDescPtr currTconst = CreateConstDesc("currT", 1, "int32");
  ge::NodePtr currTNode = graph.AddNode(currTconst);
  newNodes.push_back(currTNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "create currTNode success.");

  // build shape Node
  ge::NodePtr shapeNode = BuildShape(dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["dy"]).Clone(), dynamicGRUGradNode,
                                     graph, newNodes, failStatus);

  // build gather
  ge::GeTensorDesc totalTGatherOut(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  ge::NodePtr totalTGather =
      BuildGather("totalTGather", shapeNode->GetOpDesc()->GetOutputDesc(0).Clone(),
                  unique->GetOpDesc()->GetOutputDesc(0).Clone(), totalTGatherOut, newNodes, failStatus, graph);
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                          shapeNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(shapeNode->GetOutDataAnchor(0), totalTGather->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(unique->GetOutDataAnchor(0), totalTGather->GetInDataAnchor(1));

  // add while Node
  ge::GeTensorDesc hInputConcatDesc = ge::GeTensorDesc(GeShape(out_unique_shape), ge::FORMAT_ND, inputHType);
  ge::GeTensorDesc xInputConcatDesc = ge::GeTensorDesc(GeShape(out_unique_x_shape), ge::FORMAT_ND, inputHType);

  vector<ge::NodePtr> whileNodes = BuildWhileNodes(graph, dynamicGRUGradNode, t0HiddenGradNode, t0MatMulNode, currTNode,
                                                   totalTGather, hInputConcatDesc, xInputConcatDesc);
  int whileNodeSize = whileNodes.size();
  for (int i = 0; i < whileNodeSize; i++) {
    newNodes.push_back(whileNodes[i]);
  }
  ge::NodePtr whileNode = whileNodes[whileNodeSize - 1];
  // link to while input
  ge::GraphUtils::AddEdge(t0HiddenGradNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_2));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_3));
  ge::GraphUtils::AddEdge(t0MatMulNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(INDEX_4));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["update"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_5));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["reset"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_6));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["new"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_7));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["hidden_new"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_8));
  ge::GraphUtils::AddEdge(currTNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(INDEX_9));
  ge::GraphUtils::AddEdge(totalTGather->GetOutDataAnchor(0), whileNode->GetInDataAnchor(INDEX_10));
  ge::GraphUtils::AddEdge(gatherX->GetOutDataAnchor(0), whileNode->GetInDataAnchor(INDEX_11));
  ge::GraphUtils::AddEdge(gatherH->GetOutDataAnchor(0), whileNode->GetInDataAnchor(INDEX_12));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["weight_hidden"])->GetPeerOutAnchor(),
                          whileNode->GetInDataAnchor(INDEX_13));

  // add last cell
  ge::NodePtr tHiddenGradNode =
      AddOneHiddenGradNode(gateOrder, totalTGather, dynamicGRUGradNode, graph, newNodes, failStatus);

  // link to last cell
  ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dh_prev"]),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh_pre_t"]));
  ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(INDEX_4),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dh"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["h"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["init_h"]));
  for (InDataAnchorPtr inAnchorPtr :
       dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX["dh_prev"])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(tHiddenGradNode->GetOutDataAnchor(HIDDENGRAD_OUTPUT_INDEX["dh_prev"]), inAnchorPtr);
  }
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["dy"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["update"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["update"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["reset"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["reset"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["new"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["new"]));
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["hidden_new"])->GetPeerOutAnchor(),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["hidden_new"]));
  ge::GraphUtils::AddEdge(totalTGather->GetOutDataAnchor(0),
                          tHiddenGradNode->GetInDataAnchor(HIDDENGRAD_INPUT_INDEX["t_state"]));

  // add t0, last t and while node
  result.push_back(t0HiddenGradNode);
  result.push_back(t0MatMulNode);
  result.push_back(tHiddenGradNode);
  result.push_back(whileNode);
  result.push_back(totalTGather);

  return result;
}

map<std::string, ge::NodePtr> DynamicGRUV2GradDFusionPass::AddGRUHiddenGradNode(ge::NodePtr dynamicGRUGradNode,
                                                                                ge::ComputeGraph& graph,
                                                                                vector<ge::NodePtr>& newNodes,
                                                                                bool& failStatus) {
  map<std::string, ge::NodePtr> inputNodes;
  map<std::string, ge::NodePtr> result;
  vector<ge::NodePtr> result_node;

  // add loop t hidden grad nodes; [ [hidden_grad_nodes] [matmul_nodes] ]
  result_node = AddTLoopNode(inputNodes, dynamicGRUGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(result_node[0] == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                    return result);
  result["dgate_h"] = result_node[INDEX_3];
  result["dnt_x"] = result_node[INDEX_3];
  result["totalT"] = result_node[INDEX_4];

  ge::NodePtr dhPrevNode = result_node[INDEX_2];
  result["dh_prev"] = dhPrevNode;
  return result;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildSubNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& totalT,
                                                      ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr subDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (subDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "/GRUGrad/TSub", "Sub")),
      failStatus = true;
      return nullptr);
  subDesc->AddInputDesc("x1", totalT->GetOpDesc()->GetOutputDesc(0).Clone());

  auto subOneDesc = ge::GeTensorDesc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  subOneDesc.SetOriginShape(GeShape({1}));
  subOneDesc.SetOriginFormat(ge::FORMAT_ND);
  subDesc->AddInputDesc("x2", subOneDesc);

  subDesc->AddOutputDesc("y", subOneDesc);
  ge::GeTensorPtr subOneDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((subOneDescTensor = std::make_shared<ge::GeTensor>(subOneDesc)), failStatus = true;
                          return nullptr);
  vector<int32_t> subOneValue;
  subOneValue.push_back(static_cast<int32_t>(1));

  subOneDescTensor->SetData(reinterpret_cast<uint8_t*>(subOneValue.data()), subOneValue.size() * sizeof(int32_t));
  ge::OpDescPtr subOneOpDesc = ge::OpDescUtils::CreateConstOp(subOneDescTensor);

  ge::NodePtr subOneNode = graph.AddNode(subOneOpDesc);
  FUSION_PASS_CHECK(subOneNode == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  ge::NodePtr subNode = graph.AddNode(subDesc);
  ge::GraphUtils::AddEdge(totalT->GetOutDataAnchor(0), subNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(subOneNode->GetOutDataAnchor(0), subNode->GetInDataAnchor(1));

  return subNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildSizeConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& subNode,
                                                             ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "/GRUGrad/TConcat", "ConcatD")),
      failStatus = true;
      return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(0).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);

  concatDesc->AddInputDesc("x0", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({2}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x1", concatX2Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUM);

  ge::GeTensorPtr x2DescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((x2DescTensor = std::make_shared<ge::GeTensor>(concatX2Desc)), failStatus = true;
                          return nullptr);
  vector<int32_t> x2Value;
  x2Value.push_back(static_cast<int32_t>(batch));
  x2Value.push_back(static_cast<int32_t>(hidden_dim));

  x2DescTensor->SetData(reinterpret_cast<uint8_t*>(x2Value.data()), x2Value.size() * sizeof(int32_t));
  ge::OpDescPtr x2OpDesc = ge::OpDescUtils::CreateConstOp(x2DescTensor);

  ge::NodePtr x2Node = graph.AddNode(x2OpDesc);
  FUSION_PASS_CHECK(x2Node == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  ge::GraphUtils::AddEdge(subNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(x2Node->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddHSplitNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr& sizeConcatNode,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create slice desc
  ge::OpDescPtr sliceDesc =
      std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/Slice", "Slice");

  //
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(hidden_dim, hidden_dim));
  inputTensorDescH.SetShapeRange(x1_range);
  sliceDesc->AddInputDesc("x", inputTensorDescH);
  // build offset
  vector<int64_t> output1NZDim = inputTensorDescH.GetShape().GetDims();
  output1NZDim[0] = 0;
  output1NZDim[1] = 0;
  output1NZDim[DIM_NUM_2] = 0;
  GeTensorDesc offsetDesc(GeShape({static_cast<int64_t>(output1NZDim.size())}), ge::FORMAT_ND, DT_INT64);
  ge::OpDescPtr offsetConst = CreateListConstDesc("addHSliceOffset", output1NZDim);
  ge::NodePtr offsetNode = graph.AddNode(offsetConst);
  newNodes.push_back(offsetNode);
  sliceDesc->AddInputDesc("offsets", offsetDesc);

  // build slice size
  GeTensorDesc sizeDesc(GeShape({static_cast<int64_t>(output1NZDim.size())}), FORMAT_ND, DT_INT64);
  sliceDesc->AddInputDesc("size", sizeDesc);

  // build oputput desc
  ge::GeTensorDesc outDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone();
  vector<int64_t> output3NZDim = inputTensorDescH.GetShape().GetDims();
  output3NZDim[0] = -1;
  outDesc.SetShape(GeShape(output3NZDim));
  outDesc.SetOriginShape(GeShape(output3NZDim));
  sliceDesc->AddOutputDesc("y", outDesc);

  // add node
  vector<string> depend_names = {"offsets", "size"};
  sliceDesc->SetOpInferDepends(depend_names);
  ge::NodePtr sliceNode = graph.AddNode(sliceDesc);
  newNodes.push_back(sliceNode);

  // add edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          sliceNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(offsetNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(sizeConcatNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(INDEX_2));

  return sliceNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddHConcatNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr splitNode,
                                                        ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                        bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/HConcatD", "ConcatD")),
                          concatDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  vector<int64_t> inputDim{1, batch, hidden_dim};
  inputTensorDescInitH.SetShape(GeShape(inputDim));
  inputTensorDescInitH.SetOriginShape(GeShape(inputDim));
  concatDesc->AddInputDesc("x0", inputTensorDescInitH);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("x1", inputTensorDescSplitH);

  // output concat_h, shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDim{-1, batch, hidden_dim};

  AddOutputNodeDesc(concatDesc, "y", outputDim, ge::FORMAT_ND, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUM);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));
  return concatNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddDwhMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr hConcatNode,
                                                          ge::NodePtr gruHiddenGradNode, ge::ComputeGraph& graph,
                                                          vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwh/BatchMatmul", "BatchMatMul")),
                          matmulDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // set x1 shape range
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_dim, hidden_dim));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x2_range.insert(x2_range.begin(), std::make_pair(1, -1));
  x2_range.insert(x2_range.begin(), std::make_pair(1, -1));

  vector<int64_t> inputx1Dims = {-1, batch, hidden_dim};
  AddInputNodeDesc(matmulDesc, "x1", inputx1Dims, ge::FORMAT_ND, inputx1Dims, ge::FORMAT_ND, inputHType, x1_range);
  vector<int64_t> inputx2Dims = {-1, batch, hidden_dim * HIDDEN_NUM};
  AddInputNodeDesc(matmulDesc, "x2", inputx2Dims, ge::FORMAT_ND, inputx2Dims, ge::FORMAT_ND, inputHType, x2_range);

  vector<int64_t> outputDim{-1, hidden_dim, HIDDEN_NUM * hidden_dim};
  AddOutputNodeDesc(matmulDesc, "y", outputDim, ge::FORMAT_ND, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  // Edge gruHiddenGradNode->GetOutDataAnchor(12)
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));         // ht
  ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(INDEX_12), matmulNode->GetInDataAnchor(1));  // dgate_h

  return matmulNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddDgateHSplitNode(ge::NodePtr dynamicGRUGradNode,
                                                            ge::NodePtr gruHiddenGradNode, ge::NodePtr whileNode,
                                                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                            bool& failStatus) {
  // create slice desc
  ge::OpDescPtr sliceDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (sliceDesc = std::make_shared<ge::OpDesc>(dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dh/Slice", "Slice")),
      sliceDesc = nullptr;
      failStatus = true;
      return nullptr);

  // add input
  ge::GeTensorDesc sliceInputDesc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_12).Clone();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  sliceInputDesc.SetShapeRange(x1_range);
  sliceDesc->AddInputDesc("x", sliceInputDesc);

  // build offset
  vector<int64_t> output1NZDim = sliceInputDesc.GetShape().GetDims();
  output1NZDim[0] = 0;
  output1NZDim[1] = 0;
  output1NZDim[DIM_NUM_2] = 0;
  GeTensorDesc offsetDesc(GeShape({static_cast<int64_t>(output1NZDim.size())}), ge::FORMAT_ND, DT_INT64);
  ge::OpDescPtr offsetConst = CreateListConstDesc("addWhileHSliceOffset", output1NZDim);
  ge::NodePtr offsetNode = graph.AddNode(offsetConst);
  newNodes.push_back(offsetNode);

  // build slice size
  vector<int64_t> output2NZDim = sliceInputDesc.GetShape().GetDims();
  output2NZDim[DIM_NUM_2] = nzHiddenDim * 2 * ALIGN_16;
  ge::OpDescPtr sizeConst = CreateListConstDesc("addWhileHSliceSize", output2NZDim);
  ge::NodePtr sizeNode = graph.AddNode(sizeConst);
  newNodes.push_back(sizeNode);
  GeTensorDesc sizeDesc(GeShape({static_cast<int64_t>(output2NZDim.size())}), FORMAT_ND, DT_INT64);

  sliceDesc->AddInputDesc("offsets", offsetDesc);
  sliceDesc->AddInputDesc("size", sizeDesc);

  // build oputput desc
  ge::GeTensorDesc outDesc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_12).Clone();
  outDesc.SetShape(GeShape(output2NZDim));
  outDesc.SetOriginShape(GeShape(output2NZDim));
  outDesc.SetFormat(FORMAT_ND);
  sliceDesc->AddOutputDesc("y", outDesc);

  // add node
  vector<string> depend_names = {"offsets", "size"};
  sliceDesc->SetOpInferDepends(depend_names);
  ge::NodePtr sliceNode = graph.AddNode(sliceDesc);
  newNodes.push_back(sliceNode);

  // add edge to slice
  ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(INDEX_12), sliceNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(offsetNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(sizeNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(INDEX_2));

  return sliceNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildUnique(std::string name, vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                     ge::ComputeGraph& graph) {
  ge::OpDescPtr uniqueDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((uniqueDesc = std::make_shared<ge::OpDesc>(name, "Unique")), uniqueDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // input
  vector<int64_t> placeholder_unique_shape = {1};
  GeTensorDesc placeholder_unique_desc(GeShape(placeholder_unique_shape), FORMAT_ND, DT_INT32);
  placeholder_unique_desc.SetOriginShape(GeShape(placeholder_unique_shape));
  uniqueDesc->AddInputDesc("x", placeholder_unique_desc);

  // output
  vector<int64_t> output_unique_shape = {-1};
  GeTensorDesc output_unique_desc(GeShape(output_unique_shape), FORMAT_ND, DT_INT32);
  GeTensorDesc idx_unique_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  uniqueDesc->AddOutputDesc("y", output_unique_desc);
  uniqueDesc->AddOutputDesc("idx", idx_unique_desc);
  ge::AttrUtils::SetDataType(uniqueDesc, "out_idx", DT_INT32);

  // create node
  ge::NodePtr uniqueNode = AddNewNode(graph, uniqueDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  return uniqueNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::BuildGather(std::string name, ge::GeTensorDesc inputTensorDescH,
                                                     ge::GeTensorDesc indicesDesc, ge::GeTensorDesc outDesc,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                     ge::ComputeGraph& graph) {
  ge::OpDescPtr gatherDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((gatherDesc = std::make_shared<ge::OpDesc>(name, "Gather")), gatherDesc = nullptr;
                          failStatus = true;
                          return nullptr);
  // input
  gatherDesc->AddInputDesc("x", inputTensorDescH);
  gatherDesc->AddInputDesc("indices", indicesDesc);

  // output
  gatherDesc->AddOutputDesc("y", outDesc);

  // attr
  ge::AttrUtils::SetBool(gatherDesc, "validate_indices", true);

  // create node
  ge::NodePtr gatherNode = AddNewNode(graph, gatherDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  return gatherNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddDgateXConcatNode(ge::NodePtr dynamicGRUGradNode,
                                                             ge::NodePtr dgateHSplitNode, ge::NodePtr gruHiddenGradNode,
                                                             ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                             bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/ConcatD", "ConcatD")),
                          concatDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // input
  vector<int64_t> dirtNzDesc = {-1, batch, 2 * hidden_dim};
  vector<int64_t> dnxNzDesc = {-1, batch, hidden_dim};
  AddInputNodeDesc(concatDesc, "x0", dirtNzDesc, ge::FORMAT_ND, dirtNzDesc, ge::FORMAT_ND, inputHType);
  AddInputNodeDesc(concatDesc, "x1", dnxNzDesc, ge::FORMAT_ND, dnxNzDesc, ge::FORMAT_ND, inputHType);

  // output shape:{t,batch,3*hidden_size}
  vector<int64_t> outputDim{-1, batch, HIDDEN_NUM * hidden_dim};
  AddOutputNodeDesc(concatDesc, "y", outputDim, ge::FORMAT_ND, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", DIM_NUM_2);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUM);

  // create concat node
  ge::NodePtr concatNode = AddNewNode(graph, concatDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(dgateHSplitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));  // [dit, drt]
  ge::GraphUtils::AddEdge(gruHiddenGradNode->GetOutDataAnchor(INDEX_11), concatNode->GetInDataAnchor(1));

  return concatNode;
}

Status DynamicGRUV2GradDFusionPass::AddDxtMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                     ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create matmul desc
  ge::OpDescPtr matmulOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulOpDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dx/BatchMatmul", "BatchMatMul")),
                          matmulOpDesc = nullptr;
                          return false);

  // dgate_x
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  dgateXDesc.SetOriginShapeRange(x1_range);
  dgateXDesc.SetShapeRange(x1_range);

  // weight_x
  ge::GeTensorDesc weightXDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_input"]).Clone();
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x2_range.insert(x2_range.begin(), std::make_pair(input_dim, input_dim));
  weightXDesc.SetOriginShapeRange(x2_range);
  weightXDesc.SetShapeRange(x2_range);

  matmulOpDesc->AddInputDesc("x1", dgateXDesc);
  matmulOpDesc->AddInputDesc("x2", weightXDesc);

  // add output dxt, shape:{t, batch, input_size}
  vector<int64_t> outputDim{-1, batch, input_dim};
  AddOutputNodeDesc(matmulOpDesc, "y", outputDim, ge::FORMAT_ND, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulOpDesc, "adj_x2", true);

  // create matmul node
  bool failStatus = false;
  ge::NodePtr matmulNode = this->AddNewNode(graph, matmulOpDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return failStatus);

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

ge::NodePtr DynamicGRUV2GradDFusionPass::AddDwxMatmulNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dgateXConcatNode,
                                                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                          bool& failStatus) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicGRUGradNode->GetName() + "GRUWeightGrad/Dwx/BatchMatmul", "BatchMatMul")),
                          matmulDesc = nullptr;
                          failStatus = true;
                          return nullptr);

  // input xtDesc
  ge::GeTensorDesc xtDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).Clone();
  ge::GeTensorDesc dgateXDesc = dgateXConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();

  // set shape range {t, batch, input_dim}
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(input_dim, input_dim));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  xtDesc.SetOriginShapeRange(x1_range);
  xtDesc.SetShapeRange(x1_range);
  // set shape range  {t, batch, hiddden_dim *3}
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x2_range.insert(x2_range.begin(), std::make_pair(1, -1));
  x2_range.insert(x2_range.begin(), std::make_pair(1, -1));
  dgateXDesc.SetOriginShapeRange(x2_range);
  dgateXDesc.SetShapeRange(x2_range);

  matmulDesc->AddInputDesc("x1", xtDesc);
  matmulDesc->AddInputDesc("x2", dgateXDesc);

  // add output dwx, shape:{t, input_dim, 3 * hidden_dim}
  vector<int64_t> outputDim{-1, input_dim, HIDDEN_NUM * hidden_dim};
  AddOutputNodeDesc(matmulDesc, "y", outputDim, ge::FORMAT_ND, outputDim, ge::FORMAT_ND, inputHType);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = AddNewNode(graph, matmulDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  // input Edge
  ge::GraphUtils::AddEdge(dynamicGRUGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dgateXConcatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));
  return matmulNode;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr inputNode,
                                                          int anchorIndex, const vector<int64_t>& axis,
                                                          const string& nodeName, const string& indexName,
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
  reduceSumDesc->AddInputDesc("x", inputTensorDesc);

  // output
  ge::GeTensorDesc outputTensorDesc = dynamicGRUGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX[indexName]).Clone();
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", axis);
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = this->AddNewNode(graph, reduceSumDesc, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "check failed, fusion failed."), return nullptr);

  // Edge
  ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(anchorIndex), reduceSumNode->GetInDataAnchor(0));

  for (InDataAnchorPtr inAnchorPtr :
       dynamicGRUGradNode->GetOutDataAnchor(OUTPUT_INDEX[indexName])->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
  }
  return reduceSumNode;
}

Status DynamicGRUV2GradDFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicGRUGradNode, ge::NodePtr dwxMatmulNode,
                                                       ge::NodePtr dwhMatmulNode, ge::ComputeGraph& graph,
                                                       vector<ge::NodePtr>& newNodes) {
  // add dw_x / dw_h reduce_sum
  int anchorOutputIndex = 0;
  vector<int64_t> reduceDwAxis{0};
  bool isFailure = false;
  AddReduceSumNode(dynamicGRUGradNode, dwxMatmulNode, anchorOutputIndex, reduceDwAxis, "dwx", "dw_input", graph,
                   newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddReduceSumNode(dynamicGRUGradNode, dwhMatmulNode, anchorOutputIndex, reduceDwAxis, "dwh", "dw_hidden", graph,
                   newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

Status DynamicGRUV2GradDFusionPass::AddDbReduceSumNode(ge::NodePtr gruV2GradNode, ge::NodePtr dbxNode,
                                                       ge::NodePtr dbhNode, ge::ComputeGraph& graph,
                                                       vector<ge::NodePtr>& newNodes) {
  // add db_x / db_h reduce_sum
  int anchorOutputIndex = 12;
  bool isFailure = false;
  vector<int64_t> reduceDbAxis{0, 1};

  AddReduceSumNode(gruV2GradNode, dbxNode, 0, reduceDbAxis, "dbx", "db_input", graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDbxReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  AddReduceSumNode(gruV2GradNode, dbhNode, anchorOutputIndex, reduceDbAxis, "dbh", "db_hidden", graph, newNodes,
                   isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDbhReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  return SUCCESS;
}

vector<ge::NodePtr> DynamicGRUV2GradDFusionPass::BuildWhileNodes(ge::ComputeGraph& graph,
                                                                 ge::NodePtr dynamicGRUGradNode, ge::NodePtr t0Cell,
                                                                 ge::NodePtr t0Matmul, ge::NodePtr currTConst,
                                                                 ge::NodePtr tSizeConst, ge::GeTensorDesc concatHDesc,
                                                                 ge::GeTensorDesc concatXDesc) {
  int32_t arg_num = 14;
  OpDescBuilder op_desc_builder("Gruv2GradWhile", "While");

  // set while input desc
  OpDescPtr op_desc =
      op_desc_builder.AddInput("input0", t0Cell->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddInput("input1", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone())
          .AddInput("input2", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone())
          .AddInput("input3", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["dy"]).Clone())
          .AddInput("input4", t0Matmul->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddInput("input5", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["update"]).Clone())
          .AddInput("input6", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["reset"]).Clone())
          .AddInput("input7", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["new"]).Clone())
          .AddInput("input8", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone())
          .AddInput("input9", currTConst->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddInput("input10", tSizeConst->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddInput("input11", concatXDesc)
          .AddInput("input12", concatHDesc)
          .AddInput("input13", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone())
          .AddOutput("output0", t0Cell->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddOutput("output1", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).Clone())
          .AddOutput("output2", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone())
          .AddOutput("output3", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["dy"]).Clone())
          .AddOutput("output4", t0Matmul->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddOutput("output5", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["update"]).Clone())
          .AddOutput("output6", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["reset"]).Clone())
          .AddOutput("output7", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["new"]).Clone())
          .AddOutput("output8", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["hidden_new"]).Clone())
          .AddOutput("output9", currTConst->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddOutput("output10", tSizeConst->GetOpDesc()->GetOutputDesc(0).Clone())
          .AddOutput("output11", concatXDesc)
          .AddOutput("output12", concatHDesc)
          .AddOutput("output13", dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone())
          .Build();
  if (op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create while op_desc failed, name:Gruv2GradWhile.");
    return {};
  }
  ge::NodePtr whileNode = graph.AddNode(op_desc);
  if (whileNode == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create while node failed, name:Gruv2GradWhile.");
    return {};
  }
  // build cond graph
  ge::ComputeGraphPtr cond_graph = BuildCondGraph(whileNode, arg_num);
  if ((cond_graph == nullptr) || (graph.AddSubgraph(cond_graph) != ge::GRAPH_SUCCESS)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add  while_cond_graph node failed.");
    return {};
  }

  // build body graph
  ge::ComputeGraphPtr body_graph = BuildBodyGraph(whileNode, arg_num, dynamicGRUGradNode);
  if ((body_graph == nullptr) || (graph.AddSubgraph(body_graph) != ge::GRAPH_SUCCESS)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add  while_body_graph node failed.");
    return {};
  }

  auto graphNodes = cond_graph->GetAllNodes();
  auto bodyNodes = body_graph->GetAllNodes();
  vector<ge::NodePtr> result;
  std::copy(graphNodes.begin(), graphNodes.end(), std::back_inserter(result));
  std::copy(bodyNodes.begin(), bodyNodes.end(), std::back_inserter(result));
  result.push_back(whileNode);
  return result;
}

// build cond sub graph
ge::ComputeGraphPtr DynamicGRUV2GradDFusionPass::BuildCondGraph(ge::NodePtr whileNode, int32_t argNum) {
  string condName = "cond";
  // add parten node
  CompleteGraphBuilder graph_builder(condName, false);
  graph_builder.SetParentNode(whileNode);

  // add less node
  std::string lessName = "Less";
  OpDescBuilder op_desc_builder(lessName, "Less");
  GeTensorDesc out_desc(GeShape(), FORMAT_ND, DT_BOOL);
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(INDEX_9).Clone())
      .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(INDEX_10).Clone())
      .AddOutput("y", out_desc);
  graph_builder.AddNode(op_desc_builder.Build());

  // set input
  for (int32_t i = 0; i < INDEX_9; i++) {
    graph_builder.SetUselessInput(i);
  }
  graph_builder.SetInput(INDEX_9, {lessName}, {0});
  graph_builder.SetInput(INDEX_10, {lessName}, {1});
  for (int32_t i = INDEX_11; i < INDEX_14; i++) {
    graph_builder.SetUselessInput(i);
  }

  // describe cond output
  graph_builder.AddOutput(lessName, 0);

  // add input maping
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = 0; i < argNum; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  ge::graphStatus error_code = ge::GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr cond_graph = graph_builder.Build(error_code, error_msg);
  if (cond_graph == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Build cond_graph failed.");
    return nullptr;
  }

  // set sub graph
  size_t index = whileNode->GetOpDesc()->GetSubgraphInstanceNames().size();
  whileNode->GetOpDesc()->AddSubgraphName("cond");
  whileNode->GetOpDesc()->SetSubgraphInstanceName(index, condName);
  return cond_graph;
}

ge::OpDescPtr DynamicGRUV2GradDFusionPass::CreateConstDesc(const std::string& name, int32_t value,
                                                           const std::string& dtype) {
  OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>(name, "Const");
  if (const_op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create const desc failed. const:" + name.c_str());
    return nullptr;
  }

  GeTensorDesc data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_value =
      std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(&value), sizeof(int32_t));

  if (dtype == "int64") {
    data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64);
    const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(&value), sizeof(int64_t));
  }
  if (const_value == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create tensor failed. const:" + name.c_str());
    return nullptr;
  }

  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ATTR_NAME_WEIGHTS failed. const:" + name.c_str());
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ATTR_NAME_WEIGHTS failed. const:" + name.c_str());
    return nullptr;
  }

  return const_op_desc;
}

ge::OpDescPtr DynamicGRUV2GradDFusionPass::CreateListConstDesc(const std::string& name, std::vector<int64_t> values) {
  OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>(name, "Const");
  if (const_op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create const desc failed. const:" + name.c_str());
    return nullptr;
  }

  GeTensorDesc data_desc(GeShape({static_cast<int64_t>(values.size())}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(values.data()),
                                                           sizeof(int64_t) * values.size());
  if (const_value == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create tensor failed. const:" + name.c_str());
    return nullptr;
  }

  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ATTR_NAME_WEIGHTS failed. const:" + name.c_str());
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ATTR_NAME_WEIGHTS failed. const:" + name.c_str());
    return nullptr;
  }

  return const_op_desc;
}

ge::ComputeGraphPtr DynamicGRUV2GradDFusionPass::BuildBodyGraph(ge::NodePtr& whileNode,
                                                                int32_t argNum, ge::NodePtr dynamicGRUGradNode) {
  std::string body_name = "Body";
  // add parten node
  CompleteGraphBuilder graph_builder(body_name, false);
  graph_builder.SetParentNode(whileNode);

  // add grugradcell node
  ge::OpDescPtr dynamicGRUGradDesc = dynamicGRUGradNode->GetOpDesc();
  string gateOrder = "zrh";
  ge::AttrUtils::GetStr(dynamicGRUGradDesc, "gate_order", gateOrder);

  string cellName = dynamicGRUGradNode->GetName() + "/GRUV2Grad/GRUV2HiddenGradCell_GRUV2HiddenGradCell";
  ge::OpDescPtr hiddenGradDesc = buildCellDesc(cellName, gateOrder, whileNode);
  graph_builder.AddNode(hiddenGradDesc);

  // add dnt_x concat node
  string concat_x_name = "then_ConcatDntX_dnt_x";
  string concat_h_name = "then_ConcatDgateH_dgate_h";
  GeTensorDesc xInput0Desc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_11).Clone();
  GeTensorDesc xInput1Desc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_11).Clone();
  xInput0Desc.SetFormat(ge::FORMAT_ND);
  xInput0Desc.SetOriginFormat(ge::FORMAT_ND);
  vector<int64_t> inputDim{1, batch, hidden_dim};
  xInput0Desc.SetShape(GeShape(inputDim));
  xInput0Desc.SetOriginShape(GeShape(inputDim));
  ge::OpDescPtr dntxConcatDesc =
      buildConcatDesc(concat_x_name, dynamicGRUGradNode, xInput0Desc, xInput1Desc, xInput1Desc);
  graph_builder.AddNode(dntxConcatDesc);

  // add dgate_h concat node
  GeTensorDesc hInput0Desc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_12).Clone();
  GeTensorDesc hInput1Desc = whileNode->GetOpDesc()->GetOutputDesc(INDEX_12).Clone();
  hInput0Desc.SetFormat(ge::FORMAT_ND);
  hInput0Desc.SetOriginFormat(ge::FORMAT_ND);
  vector<int64_t> inputDimH{1, batch, hidden_dim * HIDDEN_NUM};
  hInput0Desc.SetShape(GeShape(inputDimH));
  hInput0Desc.SetOriginShape(GeShape(inputDimH));
  hInput1Desc.SetOriginShape(GeShape({-1, batch, hidden_dim * HIDDEN_NUM}));
  ge::OpDescPtr dgateHConcatDesc =
      buildConcatDesc(concat_h_name, dynamicGRUGradNode, hInput0Desc, hInput1Desc, hInput1Desc);
  graph_builder.AddNode(dgateHConcatDesc);
  // build old if wxz end
  // build add desc
  string constName = "oneConst";
  graph_builder.AddNode(CreateConstDesc(constName, 1, "int32"));
  std::string addName = "Add";
  OpDescBuilder op_desc_builder(addName, addName);
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(INDEX_9).Clone())
      .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(INDEX_9).Clone())
      .AddOutput("y", whileNode->GetOpDesc()->GetInputDesc(INDEX_9).Clone());
  graph_builder.AddNode(op_desc_builder.Build());

  // add batchmatmul node
  string batchMatMulName = "BatchMatMul";
  ge::OpDescPtr matmulDesc = AddBodyMatmulNode(batchMatMulName, hInput0Desc, dynamicGRUGradNode);
  graph_builder.AddNode(matmulDesc);

  graph_builder.SetInput(INDEX_9, {addName, cellName}, {0, 9});
  graph_builder.SetInput(0, {cellName}, {0})
      .SetInput(1, {cellName}, {1})
      .SetInput(INDEX_2, {cellName}, {8})
      .SetInput(INDEX_3, {cellName}, {2})
      .SetInput(INDEX_4, {cellName}, {3})
      .SetInput(INDEX_5, {cellName}, {4})
      .SetInput(INDEX_6, {cellName}, {5})
      .SetInput(INDEX_7, {cellName}, {6})
      .SetInput(INDEX_8, {cellName}, {7})
      .SetInput(INDEX_11, {concat_x_name}, {1})
      .SetInput(INDEX_12, {concat_h_name}, {1});
  graph_builder.SetUselessInput(INDEX_10);
  graph_builder.SetInput(INDEX_13, {batchMatMulName}, {1});

  // add link
  graph_builder.AddDataLink(cellName, 1, batchMatMulName, 0)
      .AddDataLink(constName, 0, addName, 1)
      .AddDataLink(cellName, 1, concat_h_name, 0)
      .AddDataLink(cellName, INDEX_2, concat_x_name, 0);

  // describe body output
  graph_builder.AddOutput(cellName, 0);
  for (uint32_t i = 1; i < INDEX_4; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  graph_builder.AddOutput(batchMatMulName, 0);
  for (uint32_t i = INDEX_5; i < INDEX_9; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  graph_builder.AddOutput(addName, 0);
  for (uint32_t i = INDEX_10; i < INDEX_11; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  graph_builder.AddOutput(concat_x_name, 0);
  graph_builder.AddOutput(concat_h_name, 0);
  graph_builder.AddOutput("Data_13", 0);

  // add input mapping
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = 0; i < argNum; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  // add output mapping
  std::map<uint32_t, uint32_t> output_mapping;
  for (int32_t i = 0; i < argNum; i++) {
    output_mapping[i] = i;
  }
  graph_builder.SetOutputMapping(output_mapping);

  ge::graphStatus error_code = ge::GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr body_graph = graph_builder.Build(error_code, error_msg);
  if (body_graph == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Build body_graph failed.");
    return nullptr;
  }

  // set sub graph
  size_t index = whileNode->GetOpDesc()->GetSubgraphInstanceNames().size();
  whileNode->GetOpDesc()->AddSubgraphName("Body");
  whileNode->GetOpDesc()->SetSubgraphInstanceName(index, body_name);

  return body_graph;
}

ge::OpDescPtr DynamicGRUV2GradDFusionPass::buildConcatDesc(const std::string& nodeName,
                                                           ge::NodePtr dynamicGRUGradNode, ge::GeTensorDesc input0Desc,
                                                           ge::GeTensorDesc input1Desc, ge::GeTensorDesc outputDesc) {
  // create concat desc
  ge::OpDescPtr concatDesc = std::make_shared<ge::OpDesc>(nodeName, "ConcatD");
  concatDesc->AddInputDesc("x0", input0Desc);
  concatDesc->AddInputDesc("x1", input1Desc);
  concatDesc->AddOutputDesc("y", outputDesc);

  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUM);

  return concatDesc;
}

ge::OpDescPtr DynamicGRUV2GradDFusionPass::AddBodyMatmulNode(const string& nodeName, ge::GeTensorDesc inputDesc,
                                                             ge::NodePtr dynamicGRUGradNode) {
  // create matmul desc
  ge::OpDescPtr matmulDesc = std::make_shared<ge::OpDesc>(nodeName, "BatchMatMul");
  inputDesc.SetDataType(ge::DT_FLOAT16);
  vector<int64_t> inputDim{1, batch, hidden_dim * HIDDEN_NUM};
  inputDesc.SetOriginShape(GeShape(inputDim));
  inputDesc.SetShape(GeShape(inputDim));
  inputDesc.SetFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  x1_range.insert(x1_range.begin(), std::make_pair(1, 1));
  inputDesc.SetShapeRange(x1_range);

  // weight
  ge::GeTensorDesc weightDesc = dynamicGRUGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["weight_hidden"]).Clone();
  weightDesc.SetOriginFormat(ge::FORMAT_ND);
  weightDesc.SetOriginShape(GeShape({hidden_dim, hidden_dim * HIDDEN_NUM}));
  weightDesc.SetFormat(ge::FORMAT_ND);
  weightDesc.SetShape(GeShape({hidden_dim, hidden_dim * HIDDEN_NUM}));
  weightDesc.SetDataType(inputHType);

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim * HIDDEN_NUM, hidden_dim * HIDDEN_NUM));
  x2_range.insert(x2_range.begin(), std::make_pair(hidden_dim, hidden_dim));
  weightDesc.SetShapeRange(x2_range);

  matmulDesc->AddInputDesc("x1", inputDesc);
  matmulDesc->AddInputDesc("x2", weightDesc);

  vector<int64_t> outputDim{1, batch, hidden_dim};
  AddOutputNodeDesc(matmulDesc, "y", outputDim, inputHType, ge::FORMAT_ND);

  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", true);

  return matmulDesc;
}

ge::OpDescPtr DynamicGRUV2GradDFusionPass::buildCellDesc(const string& cellName, const string& gateOrder,
                                                         ge::NodePtr& whileNode) {
  // build cell node
  ge::OpDescPtr hiddenGradDesc = std::make_shared<ge::OpDesc>(cellName, "DynamicGRUCellGrad");

  // set attr of gate  order wxz
  ge::AttrUtils::SetStr(hiddenGradDesc, "gate_order", gateOrder);

  // set cell input desc
  hiddenGradDesc->AddInputDesc("dh_pre_t", whileNode->GetOpDesc()->GetInputDesc(0).Clone());
  hiddenGradDesc->AddInputDesc("h", whileNode->GetOpDesc()->GetInputDesc(1).Clone());
  hiddenGradDesc->AddInputDesc("dy", whileNode->GetOpDesc()->GetInputDesc(INDEX_3).Clone());

  hiddenGradDesc->AddInputDesc("dh", whileNode->GetOpDesc()->GetInputDesc(INDEX_4).Clone());
  hiddenGradDesc->AddInputDesc("update", whileNode->GetOpDesc()->GetInputDesc(INDEX_5).Clone());
  hiddenGradDesc->AddInputDesc("reset", whileNode->GetOpDesc()->GetInputDesc(INDEX_6).Clone());
  hiddenGradDesc->AddInputDesc("new", whileNode->GetOpDesc()->GetInputDesc(INDEX_7).Clone());
  hiddenGradDesc->AddInputDesc("hidden_new", whileNode->GetOpDesc()->GetInputDesc(INDEX_8).Clone());
  hiddenGradDesc->AddInputDesc("init_h", whileNode->GetOpDesc()->GetInputDesc(INDEX_2).Clone());
  hiddenGradDesc->AddInputDesc("t_state", whileNode->GetOpDesc()->GetInputDesc(INDEX_10).Clone());

  hiddenGradDesc->AddOutputDesc("dh_prev", whileNode->GetOpDesc()->GetInputDesc(0).Clone());

  vector<int64_t> dgateHDim{1, batch, HIDDEN_NUM * hidden_dim};
  vector<int64_t> singleGateZDim{1, batch, hidden_dim};
  AddOutputNodeDesc(hiddenGradDesc, "dgate_h", dgateHDim, inputHType, ge::FORMAT_ND);
  AddOutputNodeDesc(hiddenGradDesc, "dnt_x", singleGateZDim, inputHType, ge::FORMAT_ND);

  return hiddenGradDesc;
}

ge::NodePtr DynamicGRUV2GradDFusionPass::AddInputReshapeNode(ge::NodePtr dynamicGRUGradNode, string reshapeName,
                                                             ge::GeTensorDesc inputDesc, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& newNodes) {
  std::string reshape_op_name = dynamicGRUGradNode->GetName() + "/" + reshapeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(reshape_op_name.c_str(), "Unsqueeze");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  // shape dims :2 ->3 , like {1, batch, hidden_dim}
  vector<int64_t> outputReshapeDims = {1, batch, hidden_dim};
  ge::GeShape outputReshapeShape(outputReshapeDims);
  ge::GeTensorDesc reshapeCellOutputDesc = ge::GeTensorDesc(outputReshapeShape, ge::FORMAT_ND, inputHType);
  reshapeCellOutputDesc.SetOriginShape(outputReshapeShape);
  reshapeCellOutputDesc.SetOriginFormat(ge::FORMAT_ND);

  // shape range
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_dim, hidden_dim));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  inputDesc.SetShapeRange(x1_range);
  inputDesc.SetOriginShapeRange(x1_range);

  // set attr
  ge::AttrUtils::SetListInt(reshape_desc, "axes", {0});

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  FUSION_PASS_CHECK(myReshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Create node error"), return nullptr);
  return myReshape_node;
}

Status DynamicGRUV2GradDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradDFusionPass fusion begin.");
  bool isFailure = false;
  // get dynamicGRUGradNode
  ge::NodePtr gruV2GradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(gruV2GradNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DynamicGRUV2Grad:grad node is null, fusion failed."),
                    return FAILED);
  ge::OpDescPtr dynamicGRUGradDesc = gruV2GradNode->GetOpDesc();
  FUSION_PASS_CHECK(dynamicGRUGradDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DynamicGRUV2Grad:op desc is null, fusion failed."), return FAILED);
  ge::GeTensorDesc inputTensorDescH = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["h"]);
  ge::GeTensorDesc inputTensorDescX = dynamicGRUGradDesc->GetInputDesc(INPUT_INDEX["x"]);

  // init shape
  this->GetNodeInfo(gruV2GradNode);

  if (PatternFusionUtil::IsUnknownShape(hidden_dim) || PatternFusionUtil::IsUnknownShape(input_dim)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "DynamicGRUV2GradDFusionPass for hidden_dim/input_dim cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (!PatternFusionUtil::IsUnknownShape(inputTensorDescH.GetShape().GetDim(0)) &&
      !PatternFusionUtil::IsUnknownShape(inputTensorDescH.GetShape().GetDim(1))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "DynamicGRUV2GradDFusionPass for t_size/batch_size cannot be applied for static shape.");
    return NOT_CHANGED;
  }

  // add reshape node ,for Change init_h from 2 dims to 3 dims.
  std::string reshapeName = "inithReshapeNode";
  ge::GeTensorDesc inputTensorDescInitH = gruV2GradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).Clone();
  ge::NodePtr reshapeInitC =
      AddInputReshapeNode(gruV2GradNode, reshapeName, inputTensorDescInitH, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddReshapeNode:check failed, fusion failed."),
                    return FAILED);
  ge::GraphUtils::AddEdge(gruV2GradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          reshapeInitC->GetInDataAnchor(0));

  // add gruHiddenGrad {dhPrevNode, dgateHConcatTNode, dntXConcatTNode}
  map<std::string, ge::NodePtr> hiddenGradNodes = AddGRUHiddenGradNode(gruV2GradNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddGRUHiddenGradNode:check failed, fusion failed."),
                    return FAILED);

  ge::NodePtr dwhMatmulNode;
  // add sub node for slice
  ge::NodePtr subNode = BuildSubNode(gruV2GradNode, hiddenGradNodes["totalT"], graph, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "Add totalT:check failed, fusion failed."),
                    return FAILED);
  // add concat t for slice size.
  ge::NodePtr sizeConcatNode = BuildSizeConcatNode(gruV2GradNode, subNode, graph, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat t for slice:check failed, fusion failed."),
                    return FAILED);
  // add split
  ge::NodePtr splitNode = AddHSplitNode(gruV2GradNode, sizeConcatNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // add concat
  ge::NodePtr hConcatNode = AddHConcatNode(gruV2GradNode, splitNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHConcatNode:check failed, fusion failed."),
                    return FAILED);
  ge::GraphUtils::AddEdge(reshapeInitC->GetOutDataAnchor(0), hConcatNode->GetInDataAnchor(0));

  // add dw_h : matmul(h_prev.T, dgate_h)
  dwhMatmulNode = AddDwhMatmulNode(gruV2GradNode, hConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwhMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // split dgate_h to [dit, drt] and [dnt_h]
  ge::NodePtr dgateHSplitNode = nullptr;
  dgateHSplitNode = AddDgateHSplitNode(gruV2GradNode, hiddenGradNodes["dgate_h"], hiddenGradNodes["dgate_h"], graph,
                                       newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDgateHSplitNode:check failed, fusion failed."),
                    return FAILED);

  // concat [dit, drt] with [dnt_x] to dgate_x
  ge::NodePtr gateConcatNode = nullptr;
  gateConcatNode =
      AddDgateXConcatNode(gruV2GradNode, dgateHSplitNode, hiddenGradNodes["dnt_x"], graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDgateXConcatNode:check failed, fusion failed."),
                    return FAILED);

  // add dxt matmul(dgate_x, w_x.T)
  isFailure = AddDxtMatmulNode(gruV2GradNode, gateConcatNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDxtMatmulNode:check failed, fusion failed."),
                    return FAILED);

  // add dw_x matmul(x.T, dgate_x)
  ge::NodePtr dwxMatmulNode = AddDwxMatmulNode(gruV2GradNode, gateConcatNode, graph, newNodes, isFailure);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwxMatmulNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddDwReduceSumNode(gruV2GradNode, dwxMatmulNode, dwhMatmulNode, graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDwReduceSumNode:check failed, fusion failed."),
                    return FAILED);

  isFailure = AddDbReduceSumNode(gruV2GradNode, gateConcatNode, hiddenGradNodes["dgate_h"], graph, newNodes);
  FUSION_PASS_CHECK(isFailure, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddDbReduceSumNode:check failed, fusion failed."),
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

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicGRUV2GradDFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("DynamicGRUV2GradDFusionPass", BUILT_IN_GRAPH_PASS, DynamicGRUV2GradDFusionPass);
}  // namespace fe