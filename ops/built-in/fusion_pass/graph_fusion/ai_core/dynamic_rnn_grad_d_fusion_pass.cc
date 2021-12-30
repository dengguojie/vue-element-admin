/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_grad_d_fusion_pass.h"

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
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {

static const int X_INDEX = 0;
static const int W_INDEX = 1;
static const int INIT_C_INDEX = 5;
static const int INIT_H_INDEX = 4;
static const int H_INDEX = 6;
static const int C_INDEX = 7;
static const int DY_INDEX = 8;
static const int DH_INDEX = 9;
static const int DC_INDEX = 10;
static const int I_INDEX = 11;
static const int J_INDEX = 12;
static const int F_INDEX = 13;
static const int O_INDEX = 14;
static const int TANHCT_INDEX = 15;
static const int MASK_INDEX = 16;
static const int BLOCKSIZE = 16;
static const char* FUSED_NODE = "DynamicRNNGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNGrad";
static map<std::string, int> RNN_GRAD_NODE_INPUT_INDEX = {{"x", X_INDEX}, {"w", W_INDEX}, {"init_c", INIT_C_INDEX}, {"init_h", INIT_H_INDEX}, {"h", H_INDEX},
                                                          {"c", C_INDEX}, {"dy", DY_INDEX}, {"dh", DH_INDEX}, {"dc", DC_INDEX}, {"i", I_INDEX},
                                                          {"j", J_INDEX}, {"f", F_INDEX}, {"o", O_INDEX}, {"tanhct", TANHCT_INDEX},
                                                          {"mask", MASK_INDEX}};

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

vector<FusionPattern*> DynamicRNNGradDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNGradDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}


ge::ComputeGraphPtr DynamicRNNGradDFusionPass::BuildCondGraph(ge::NodePtr &whileNode, int32_t argNum)
{
  string condName = DynamicRNNGradName + "cond";
  CompleteGraphBuilder graph_builder(condName, false);
  graph_builder.SetParentNode(whileNode);
  std::string lessName = DynamicRNNGradName + "Less";
  OpDescBuilder op_desc_builder(lessName, "Less");
  GeTensorDesc out_desc(GeShape(), FORMAT_ND, DT_BOOL);
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(O_INDEX).Clone())
                 .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(TANHCT_INDEX).Clone())
                 .AddOutput("y", out_desc);
  graph_builder.AddNode(op_desc_builder.Build());
  for (int32_t i = X_INDEX; i < O_INDEX; i++) {
    graph_builder.SetUselessInput(i);
  }
  graph_builder.SetInput(O_INDEX, {lessName}, {X_INDEX});
  graph_builder.SetInput(TANHCT_INDEX, {lessName}, {W_INDEX});
  graph_builder.SetUselessInput(MASK_INDEX);
  graph_builder.SetUselessInput(17);
  graph_builder.AddOutput(lessName, X_INDEX);
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = X_INDEX; i < argNum; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);
  ge::graphStatus error_code = ge::GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr cond_graph = graph_builder.Build(error_code, error_msg);
  if (cond_graph == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Build cond_graph failed: error_code:%u.", error_msg.c_str());
    return nullptr;
  }
  size_t index = whileNode->GetOpDesc()->GetSubgraphInstanceNames().size();
  whileNode->GetOpDesc()->AddSubgraphName(DynamicRNNGradName + "cond");
  whileNode->GetOpDesc()->SetSubgraphInstanceName(index, condName);
  return cond_graph;
}

ge::OpDescPtr DynamicRNNGradDFusionPass::CreateListConstDesc(const std::string &name, std::vector<int64_t> values) {
  OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")),
       return nullptr);

  GeTensorDesc data_desc(GeShape({static_cast<int64_t>(values.size())}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t *>(values.data()),
                                                     sizeof(int64_t) * values.size())),
       return nullptr);
  if (const_value == nullptr) {
    return nullptr;
  }
  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  return const_op_desc;
}

ge::OpDescPtr DynamicRNNGradDFusionPass::CreateConstDesc(const std::string &name, int32_t value,
                                                         const std::string &dtype) {
  OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")),
       return nullptr);
  if (const_op_desc == nullptr) {
    return nullptr;
  }
  GeTensorDesc data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value),
                                                           sizeof(int32_t))),
       return nullptr);
  if (dtype == "int64") {
    data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64);
    FUSION_PASS_MAKE_SHARED(
      (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value),
                                                 sizeof(int64_t))),
       return nullptr);
  }

  if (const_value == nullptr) {
    return nullptr;
  }
  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  return const_op_desc;
}

ge::NodePtr DynamicRNNGradDFusionPass::GetDynamicReshapeDxNode(std::string& reshapeNodeName,
                                                               ge::NodePtr dynamicRNNGradNode,
                                                               ge::GeTensorDesc inputDesc, ge::GeTensorDesc outputDesc,
                                                               ge::ComputeGraph& graph, bool& failStatus){
  std::string operatorName = dynamicRNNGradNode->GetName() + "/" + reshapeNodeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(),
                                                       "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  ge::GeShape inputShape = ge::GeShape({(inputDesc.GetOriginShape().GetDim(W_INDEX) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX});
  inputDesc.SetShape(inputShape);
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                           "Reshape node update outputDesc failed!"),
                    return nullptr);
  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeShape outputShape = ge::GeShape({-W_INDEX, (inputDesc.GetOriginShape().GetDim(W_INDEX) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX});
  ge::GeShape outputOriShape = ge::GeShape({-W_INDEX, batch_size, inputDesc.GetOriginShape().GetDim(W_INDEX)});
  auto outDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outDesc.SetOriginShape(outputOriShape);
  outDesc.SetOriginFormat(ge::FORMAT_ND);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                  "Reshape node update outputDesc failed!"),
                    return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                  "Reshape node update outputDesc failed!"),
                    return nullptr);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  return myReshape_node;
}

vector<ge::OpDescPtr> DynamicRNNGradDFusionPass::GetDynamicBodyReshapeNode(std::string& reshapeNodeName,
                                     std::string& reshapeConstNodeName, ge::NodePtr dynamicRNNGradNode,
                                     ge::GeTensorDesc inputDesc, ge::GeTensorDesc outputDesc,
                                                                 ge::ComputeGraph& graph, bool& failStatus){
  vector<ge::OpDescPtr> res = {};
  auto reshapeOp = ge::OperatorFactory::CreateOperator(reshapeNodeName.c_str(), "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return res);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                     "Reshape node update outputDesc failed!"),
                    return res);

  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                        "Reshape node update outputDesc failed!"),
                    return res);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                        "Reshape node update outputDesc failed!"),
                    return res);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  int64_t dim1 = W_INDEX;
  ge::OpDescPtr const_opdesc = CreateListConstDesc(reshapeConstNodeName, {dim1, MASK_INDEX, INIT_H_INDEX*hidden_size});
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op operator error"), return res);
  return {reshape_desc, const_opdesc};
}

vector<ge::OpDescPtr> DynamicRNNGradDFusionPass::GetDynamicBodyDxReshapeNode(std::string& reshapeNodeName,
                        std::string& reshapeConstNodeName, ge::NodePtr dynamicRNNGradNode, ge::GeTensorDesc inputDesc,
                        ge::GeTensorDesc outputDesc,
                                                                 ge::ComputeGraph& graph, bool& failStatus){
  vector<ge::OpDescPtr> res = {};
  auto reshapeOp = ge::OperatorFactory::CreateOperator(reshapeNodeName.c_str(), "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return res);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                       "Reshape node update outputDesc failed!"),
                    return res);

  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                        "Reshape node update outputDesc failed!"),
                    return res);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                        "Reshape node update outputDesc failed!"),
                    return res);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  int64_t dim1 = W_INDEX;
  ge::OpDescPtr const_opdesc = CreateListConstDesc(reshapeConstNodeName, {dim1, MASK_INDEX, input_size});
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op operator error"), return res);
  return {reshape_desc, const_opdesc};
}


vector<ge::NodePtr> DynamicRNNGradDFusionPass::GetDynamicReshapeNode(std::string& reshapeNodeName,
                                                                     ge::NodePtr dynamicRNNGradNode,
                                                                     ge::NodePtr dgateInput,
                                                                     ge::GeTensorDesc outputDesc,
                                                                     ge::NodePtr shapeNode,
                                                                     ge::ComputeGraph& graph, bool& failStatus){
  vector<ge::NodePtr> result = {};
  std::string operatorName = dynamicRNNGradNode->GetName() + "/" + reshapeNodeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(),
                                                       "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return result);
  ge::GeTensorDesc inputDesc = dgateInput->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  ge::GeShape inputShape = ge::GeShape({(inputDesc.GetOriginShape().GetDim(W_INDEX) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX});
  inputDesc.SetShape(inputShape);
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                  "Reshape node update outputDesc failed!"),
                    return result);
  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeShape outputShape = ge::GeShape({-W_INDEX, (inputDesc.GetOriginShape().GetDim(W_INDEX) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX});
  ge::GeShape outputOriShape = ge::GeShape({-W_INDEX, batch_size, inputDesc.GetOriginShape().GetDim(W_INDEX)});
  auto outDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outDesc.SetOriginShape(outputOriShape);
  outDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                   "Reshape node update outputDesc failed!"),
                    return result);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                   "Reshape node update outputDesc failed!"),
                    return result);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  ge::GeShape shapeConstShape = ge::GeShape({W_INDEX,});
  auto shapeDescConst = ge::GeTensorDesc(shapeConstShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDescConst.SetOriginShape(shapeConstShape);
  shapeDescConst.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorPtr shapeDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (shapeDescTensor = std::make_shared<ge::GeTensor>(shapeDescConst)),
       return result);
  vector<int64_t> shapeValue;
  int64_t dim1 = -W_INDEX;
  shapeValue.push_back(static_cast<int64_t>(dim1));

  shapeDescTensor->SetData(reinterpret_cast<uint8_t *>(shapeValue.data()), shapeValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(shapeDescTensor);
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op desc error"), return result);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);
  FUSION_PASS_CHECK(const_node==nullptr, OP_LOGE("Create const Op operator error"), return result);
  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(myReshape_node==nullptr, OP_LOGE("Create Reshape Op operator error"), return result);
  // add const for body dgate
  ge::GeTensorPtr shapeBodyDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (shapeBodyDescTensor = std::make_shared<ge::GeTensor>(shapeDescConst)),
       return result);
  vector<int64_t> shapeBodyValue;
  shapeBodyValue.push_back(static_cast<int64_t>(W_INDEX));

  shapeBodyDescTensor->SetData(reinterpret_cast<uint8_t *>(shapeBodyValue.data()),
                                                           shapeBodyValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_body_opdesc = ge::OpDescUtils::CreateConstOp(shapeBodyDescTensor);
  ge::NodePtr const_body_node = graph.AddNode(const_body_opdesc);
  FUSION_PASS_CHECK(const_body_node==nullptr, OP_LOGE("Create const Op operator error"), return result);

  ge::GeShape lastConstShape = ge::GeShape({W_INDEX,});
  auto lastDescConst = ge::GeTensorDesc(lastConstShape, ge::FORMAT_ND, ge::DT_INT64);
  lastDescConst.SetOriginShape(lastConstShape);
  lastDescConst.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorPtr lastDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (lastDescTensor = std::make_shared<ge::GeTensor>(lastDescConst)),
        return result);
  vector<int64_t> lastValue;
  int64_t dimLast = INIT_H_INDEX*hidden_size;
  lastValue.push_back(static_cast<int64_t>(dimLast));

  lastDescTensor->SetData(reinterpret_cast<uint8_t *>(lastValue.data()), lastValue.size() * sizeof(int64_t));
  ge::OpDescPtr last_const_opdesc = ge::OpDescUtils::CreateConstOp(lastDescTensor);
  ge::NodePtr last_const_node = graph.AddNode(last_const_opdesc);
  FUSION_PASS_CHECK(last_const_node == nullptr, OP_LOGE("Create Const Op operator error"), return result);
  ge::NodePtr tSplitNode = BuildTDgateSplit(shapeNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone(),
                                            dynamicRNNGradNode, graph, failStatus);
  std::string reshapeConcatName = "TDgateConcat";
  ge::NodePtr dxReshapeConcatNode = BuildDgateReshapeSizeConcatNode(dynamicRNNGradNode,
                                                                    reshapeConcatName, const_node, graph, failStatus);
  std::string reshapeBodyConcatName = "TDgateBodyConcat";
  ge::NodePtr dgateBodyReshapeConcatNode = BuildDgateReshapeSizeConcatNode(dynamicRNNGradNode,
                                                                           reshapeBodyConcatName,
                                                                           const_body_node, graph, failStatus);
  ge::GraphUtils::AddEdge(shapeNode->GetOutDataAnchor(X_INDEX), tSplitNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(X_INDEX), dxReshapeConcatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(W_INDEX), dxReshapeConcatNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(last_const_node->GetOutDataAnchor(X_INDEX), dxReshapeConcatNode->GetInDataAnchor(2));

  ge::GraphUtils::AddEdge(dxReshapeConcatNode->GetOutDataAnchor(X_INDEX), myReshape_node->GetInDataAnchor(W_INDEX));

  ge::GraphUtils::AddEdge(const_body_node->GetOutDataAnchor(X_INDEX), dgateBodyReshapeConcatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(W_INDEX), dgateBodyReshapeConcatNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(last_const_node->GetOutDataAnchor(X_INDEX), dgateBodyReshapeConcatNode->GetInDataAnchor(2));
  return {myReshape_node, dgateBodyReshapeConcatNode};
}

ge::OpDescPtr GetDynamicBodyLSTMGradCellNode(std::string cellNodeName, ge::NodePtr dynamicRNNGradNode,
                                                                    ge::GeTensorDesc curTDesc,
                                                                    ge::ComputeGraph& graph, bool& failStatus)
{
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::GeTensorDesc inputI = dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]);
  ge::OpDescPtr basicLstmCellStateGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((basicLstmCellStateGradDesc = std::make_shared<ge::OpDesc>(cellNodeName,
                                                                                     "DynamicLSTMGradCell")),
                           failStatus=true; return nullptr);
  basicLstmCellStateGradDesc->AddInputDesc("init_c",
                                       dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]));

  basicLstmCellStateGradDesc->AddInputDesc("c",
                                       dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["c"]));
  basicLstmCellStateGradDesc->AddInputDesc("dy", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dy"]));
  basicLstmCellStateGradDesc->AddInputDesc("dh", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]));
  basicLstmCellStateGradDesc->AddInputDesc("dc", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]));

  basicLstmCellStateGradDesc->AddInputDesc("i", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]));
  basicLstmCellStateGradDesc->AddInputDesc("j", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["j"]));
  basicLstmCellStateGradDesc->AddInputDesc("f", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["f"]));
  basicLstmCellStateGradDesc->AddInputDesc("o", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["o"]));
  basicLstmCellStateGradDesc->AddInputDesc("tanhct",
                                           dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["tanhct"]));

  basicLstmCellStateGradDesc->AddInputDesc("mask", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]));

  basicLstmCellStateGradDesc->AddInputDesc("t_state", curTDesc);

  vector<int64_t> output_dims;
  output_dims.push_back(W_INDEX);
  output_dims.push_back(batch_size);
  output_dims.push_back(INIT_H_INDEX * hidden_size);

  ge::GeShape output_origin_shape(output_dims);
  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc output_tensor_desc = ge::GeTensorDesc(output_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  output_tensor_desc.SetOriginShape(output_origin_shape);
  output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dgate", output_tensor_desc);

  vector<int64_t> dc_prev_tensor_dims;
  dc_prev_tensor_dims.push_back(batch_size);
  dc_prev_tensor_dims.push_back(hidden_size);

  ge::GeShape dc_prev_tensor_shape(dc_prev_tensor_dims);
  ge::GeTensorDesc tensor_dc_prew = ge::GeTensorDesc(dc_prev_tensor_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_dc_prew.SetOriginShape(dc_prev_tensor_shape);
  tensor_dc_prew.SetOriginFormat(ge::FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dct_1", tensor_dc_prew);
  ge::AttrUtils::SetFloat(basicLstmCellStateGradDesc, "forget_bias", 1.0);
  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradDesc, "direction", direction);
  ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "direction", direction);

  std::string gate_order = "ijfo";
  ge::AttrUtils::GetStr(dynamicRNNGradDesc, "gate_order", gate_order);
  ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "gate_order", gate_order);

  return basicLstmCellStateGradDesc;

}

ge::OpDescPtr DynamicRNNGradDFusionPass::GetDynamicLSTMGradCellNode(std::string cellNodeName,
                                                                    ge::NodePtr dynamicRNNGradNode,
                                                                    ge::GeTensorDesc curTDesc,
                                                                    ge::ComputeGraph& graph, bool& failStatus)
{
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::GeTensorDesc inputI = dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]);
  ge::OpDescPtr basicLstmCellStateGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((basicLstmCellStateGradDesc = std::make_shared<ge::OpDesc>(cellNodeName,
                           "DynamicLSTMGradCell")),
                           failStatus=true; return nullptr);
  basicLstmCellStateGradDesc->AddInputDesc("init_c",
                                   dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]));

  basicLstmCellStateGradDesc->AddInputDesc("c", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["c"]));
  basicLstmCellStateGradDesc->AddInputDesc("dy", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dy"]));
  basicLstmCellStateGradDesc->AddInputDesc("dh", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]));
  basicLstmCellStateGradDesc->AddInputDesc("dc", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]));

  basicLstmCellStateGradDesc->AddInputDesc("i", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]));
  basicLstmCellStateGradDesc->AddInputDesc("j", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["j"]));
  basicLstmCellStateGradDesc->AddInputDesc("f", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["f"]));
  basicLstmCellStateGradDesc->AddInputDesc("o", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["o"]));
  basicLstmCellStateGradDesc->AddInputDesc("tanhct",
                                            dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["tanhct"]));

  basicLstmCellStateGradDesc->AddInputDesc("t_state", curTDesc);

  basicLstmCellStateGradDesc->AddInputDesc("mask",
                                           dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]));

  vector<int64_t> output_dims;
  output_dims.push_back(batch_size);
  output_dims.push_back(INIT_H_INDEX * hidden_size);

  ge::GeShape output_origin_shape(output_dims);
  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc output_tensor_desc = ge::GeTensorDesc(output_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  output_tensor_desc.SetOriginShape(output_origin_shape);
  output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dgate", output_tensor_desc);

  vector<int64_t> dc_prev_tensor_dims;
  dc_prev_tensor_dims.push_back(batch_size);
  dc_prev_tensor_dims.push_back(hidden_size);

  ge::GeShape dc_prev_tensor_shape(dc_prev_tensor_dims);
  ge::GeTensorDesc tensor_dc_prew = ge::GeTensorDesc(dc_prev_tensor_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_dc_prew.SetOriginShape(dc_prev_tensor_shape);
  tensor_dc_prew.SetOriginFormat(ge::FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dct_1", tensor_dc_prew);
  ge::AttrUtils::SetFloat(basicLstmCellStateGradDesc, "forget_bias", 1.0);
  ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "activation", "Tanh");
  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradDesc, "direction", direction);
  ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "direction", direction);

  std::string gate_order = "ijfo";
  ge::AttrUtils::GetStr(dynamicRNNGradDesc, "gate_order", gate_order);
  ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "gate_order", gate_order);

  return basicLstmCellStateGradDesc;

}

ge::OpDescPtr DynamicRNNGradDFusionPass::GetDynamicMatMulNode(std::string matmulNodeName,
                                            ge::NodePtr dynamicRNNGradNode,
                                            ge::ComputeGraph& graph, bool& failStatus, ge::GeShape dgateShape)
{
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::OpDescPtr lstmBatchMatMulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
   (lstmBatchMatMulDesc = std::make_shared<ge::OpDesc>(matmulNodeName,
                                                      "MatMulV2")),
    failStatus=true; return nullptr);
  // add matmul input
  ge::GeTensorDesc left_tensor_desc = ge::GeTensorDesc(dgateShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  left_tensor_desc.SetOriginShape(dgateShape);
  left_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(W_INDEX, 1000));
  x1_range.push_back(std::make_pair(dgateShape.GetDim(W_INDEX), dgateShape.GetDim(W_INDEX)));
  left_tensor_desc.SetShapeRange(x1_range);
  lstmBatchMatMulDesc->AddInputDesc("x1", left_tensor_desc);

  ge::GeTensorDesc w_tensor_desc = dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["w"]).Clone();
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::make_pair(w_tensor_desc.GetShape().GetDim(X_INDEX), w_tensor_desc.GetShape().GetDim(X_INDEX)));
  x2_range.push_back(std::make_pair(w_tensor_desc.GetShape().GetDim(W_INDEX), w_tensor_desc.GetShape().GetDim(W_INDEX)));
  w_tensor_desc.SetShapeRange(x2_range);
  lstmBatchMatMulDesc->AddInputDesc("x2", w_tensor_desc);
  // add matmul output
  vector<int64_t> outputy_dims;
  outputy_dims.push_back(dgateShape.GetDim(X_INDEX));
  outputy_dims.push_back(w_tensor_desc.GetShape().GetDim(X_INDEX));

  ge::GeShape outputy_origin_shape(outputy_dims);
  ge::GeShape outputy_shape(outputy_dims);
  ge::GeTensorDesc outputy_tensor_desc = ge::GeTensorDesc(outputy_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  outputy_tensor_desc.SetOriginShape(outputy_origin_shape);
  outputy_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> y1_range;
  y1_range.push_back(std::make_pair(W_INDEX, 1000));
  y1_range.push_back(std::make_pair(w_tensor_desc.GetShape().GetDim(X_INDEX), w_tensor_desc.GetShape().GetDim(X_INDEX)));
  outputy_tensor_desc.SetShapeRange(y1_range);
  lstmBatchMatMulDesc->AddOutputDesc("y", outputy_tensor_desc);
  // attr
  ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x1", false);
  ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x2", true);

  return lstmBatchMatMulDesc;
}

vector<ge::OpDescPtr> GetDynamicBodySplitNode(std::string splitNodeName, std::string splitDimNodeName,
                                              std::string splitSizeNodeName,
                                              ge::NodePtr dynamicRNNGradNode,
                                              ge::ComputeGraph& graph, bool& failStatus,
                                              ge::GeTensorDesc matmulOutputDesc){
  ge::OpDescPtr lstmSplitDesc = nullptr;
  vector<ge::OpDescPtr> nullResult = {};
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(splitNodeName, "SplitV")),
      failStatus=true; return nullResult);

  lstmSplitDesc->AddInputDesc("x", matmulOutputDesc);

  vector<int64_t> dx_dims;
  dx_dims.push_back(W_INDEX);
  dx_dims.push_back(batch_size);
  dx_dims.push_back(input_size);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y0", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back(batch_size);
  dh_dims.push_back(hidden_size);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::GeShape dh_ori_shape(dh_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y1", dh_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(input_size);
  size_splits.push_back(hidden_size);

  ge::GeShape sizeSplitShape = ge::GeShape({2});
  auto sizeSplitDesc = ge::GeTensorDesc(sizeSplitShape, ge::FORMAT_ND, ge::DT_INT32);
  sizeSplitDesc.SetOriginShape(sizeSplitShape);
  sizeSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("size_splits", sizeSplitDesc);

  ge::GeTensorPtr sizeSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (sizeSplitDescTensor = std::make_shared<ge::GeTensor>(sizeSplitDesc)),
        return nullResult);
  vector<int32_t> sizeSplitValue;
  sizeSplitValue.push_back(static_cast<int32_t>(input_size));
  sizeSplitValue.push_back(static_cast<int32_t>(hidden_size));

  sizeSplitDescTensor->SetData(reinterpret_cast<uint8_t *>(sizeSplitValue.data()),
                               sizeSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr sizeSplitOpDesc = ge::OpDescUtils::CreateConstOp(sizeSplitDescTensor);
  sizeSplitOpDesc->SetName(splitSizeNodeName);


  ge::GeShape sizeDimShape = ge::GeShape({W_INDEX});
  auto dimSplitDesc = ge::GeTensorDesc(sizeDimShape, ge::FORMAT_ND, ge::DT_INT32);
  dimSplitDesc.SetOriginShape(sizeDimShape);
  dimSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("split_dim", dimSplitDesc);

  ge::GeTensorPtr dimSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (dimSplitDescTensor = std::make_shared<ge::GeTensor>(dimSplitDesc)),
        return nullResult);
  vector<int32_t> dimSplitValue;
  dimSplitValue.push_back(static_cast<int32_t>(W_INDEX));

  dimSplitDescTensor->SetData(reinterpret_cast<uint8_t *>(dimSplitValue.data()),
                              dimSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr dimSplitOpDesc = ge::OpDescUtils::CreateConstOp(dimSplitDescTensor);
  dimSplitOpDesc->SetName(splitDimNodeName);

  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);
  vector<string> depend_names = {"size_splits", "split_dim"};
  lstmSplitDesc->SetOpInferDepends(depend_names);
  vector<ge::OpDescPtr> result = {lstmSplitDesc, sizeSplitOpDesc, dimSplitOpDesc};

  return result;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildSizeConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                           ge::NodePtr& subNode, ge::ComputeGraph& graph,
                                                           bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TConcat", "ConcatD")),
        failStatus=true; return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);

  concatDesc->AddInputDesc("x1", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({2}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x2", concatX2Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::GeTensorPtr x2DescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (x2DescTensor = std::make_shared<ge::GeTensor>(concatX2Desc)),
        failStatus=true; return nullptr);
  vector<int32_t> x2Value;
  x2Value.push_back(static_cast<int32_t>(batch_size));
  x2Value.push_back(static_cast<int32_t>(hidden_size));

  x2DescTensor->SetData(reinterpret_cast<uint8_t *>(x2Value.data()), x2Value.size() * sizeof(int32_t));
  ge::OpDescPtr x2OpDesc = ge::OpDescUtils::CreateConstOp(x2DescTensor);

  ge::NodePtr x2Node = graph.AddNode(x2OpDesc);
  FUSION_PASS_CHECK(x2Node == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  ge::GraphUtils::AddEdge(subNode->GetOutDataAnchor(X_INDEX), concatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(x2Node->GetOutDataAnchor(X_INDEX), concatNode->GetInDataAnchor(W_INDEX));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildDxReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                                    std::string& nodeName, ge::NodePtr& subNode,
                                                                    ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/" + nodeName, "ConcatD")),
        failStatus=true; return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);

  concatDesc->AddInputDesc("x1", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({2}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x2", concatX2Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildDgateReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                                       std::string& nodeName, ge::NodePtr& subNode,
                                                                       ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/" + nodeName, "ConcatD")),
        failStatus=true; return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);

  concatDesc->AddInputDesc("x1", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({W_INDEX}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({W_INDEX}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x2", concatX2Desc);

  auto concatX3Desc = ge::GeTensorDesc(GeShape({W_INDEX}), ge::FORMAT_ND, ge::DT_INT32);
  concatX3Desc.SetOriginShape(GeShape({W_INDEX}));
  concatX3Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x3", concatX3Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(concatDesc, "N", 3);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildSubNode(ge::NodePtr dynamicRNNGradNode,
                                                    ge::NodePtr& tSplitNode, ge::ComputeGraph& graph,
                                                    bool& failStatus) {
  ge::OpDescPtr subDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (subDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TSub", "Sub")),
        failStatus=true; return nullptr);
  subDesc->AddInputDesc("x1", tSplitNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone());

  auto subOneDesc = ge::GeTensorDesc(GeShape({W_INDEX}), ge::FORMAT_ND, ge::DT_INT32);
  subOneDesc.SetOriginShape(GeShape({W_INDEX}));
  subOneDesc.SetOriginFormat(ge::FORMAT_ND);
  subDesc->AddInputDesc("x2", subOneDesc);

  subDesc->AddOutputDesc("y", subOneDesc);
  ge::GeTensorPtr subOneDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (subOneDescTensor = std::make_shared<ge::GeTensor>(subOneDesc)),
        failStatus=true; return nullptr);
  vector<int32_t> subOneValue;
  subOneValue.push_back(static_cast<int32_t>(W_INDEX));

  subOneDescTensor->SetData(reinterpret_cast<uint8_t *>(subOneValue.data()), subOneValue.size() * sizeof(int32_t));
  ge::OpDescPtr subOneOpDesc = ge::OpDescUtils::CreateConstOp(subOneDescTensor);

  ge::NodePtr subOneNode = graph.AddNode(subOneOpDesc);
  FUSION_PASS_CHECK(subOneNode == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  ge::NodePtr subNode = graph.AddNode(subDesc);
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(X_INDEX), subNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(subOneNode->GetOutDataAnchor(X_INDEX), subNode->GetInDataAnchor(W_INDEX));

  return subNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildTShape(ge::GeTensorDesc xDesc, ge::NodePtr dynamicRNNGradNode,
                                                   ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr shapeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (shapeDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TShape", "Shape")),
        failStatus=true; return nullptr);
  shapeDesc->AddInputDesc("x", xDesc);
  vector<int64_t> inputDims = xDesc.GetShape().GetDims();
  ge::GeTensorDesc outputDesc = ge::GeTensorDesc(GeShape(inputDims), ge::FORMAT_ND, ge::DT_INT32);
  outputDesc.SetOriginShape(GeShape(inputDims));
  outputDesc.SetOriginFormat(ge::FORMAT_ND);
  shapeDesc->AddOutputDesc("y", outputDesc);
  ge::NodePtr shapeNode = graph.AddNode(shapeDesc);

  return shapeNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildTDgateSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode,
                                                        ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr lstmSplitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TDgateSplit", "SplitVD")),
      failStatus=true; return nullptr);
  lstmSplitDesc->AddInputDesc("y", shapeDesc);
  vector<int64_t> dx_dims;
  dx_dims.push_back(W_INDEX);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_ND, ge::DT_INT32);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("t", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back(W_INDEX);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape dh_ori_shape(dh_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

  vector<int64_t> last_dims;
  last_dims.push_back(W_INDEX);
  ge::GeShape last_shape(last_dims);
  ge::GeTensorDesc last_tensor_desc = ge::GeTensorDesc(last_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape last_ori_shape(last_dims);

  last_tensor_desc.SetOriginShape(last_ori_shape);
  last_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("last_dim", last_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(W_INDEX);
  size_splits.push_back(W_INDEX);
  size_splits.push_back(W_INDEX);
  ge::AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(lstmSplitDesc, "split_dim", X_INDEX);
  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 3);

  ge::NodePtr splitNode = graph.AddNode(lstmSplitDesc);

  return splitNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::BuildTSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode,
                                                   ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr lstmSplitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TSplit", "SplitVD")),
      failStatus=true; return nullptr);
  lstmSplitDesc->AddInputDesc("y", shapeDesc);
  vector<int64_t> dx_dims;
  dx_dims.push_back(W_INDEX);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_ND, ge::DT_INT32);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("t", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back(shapeDesc.GetShape().GetDim(X_INDEX) - W_INDEX);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape dh_ori_shape(dh_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(W_INDEX);
  size_splits.push_back(shapeDesc.GetShape().GetDim(X_INDEX) - W_INDEX);
  ge::AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(lstmSplitDesc, "split_dim", X_INDEX);
  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);

  ge::NodePtr splitNode = graph.AddNode(lstmSplitDesc);

  return splitNode;
}
vector<ge::OpDescPtr> DynamicRNNGradDFusionPass::GetDynamicSplitNode(std::string splitNodeName,
                                                                     std::string splitDimNodeName,
                                                                     std::string splitSizeNodeName,
                                                                   ge::NodePtr dynamicRNNGradNode,
                                                                   ge::ComputeGraph& graph, bool& failStatus,
                                                                   ge::GeTensorDesc matmulOutputDesc){
  ge::OpDescPtr lstmSplitDesc = nullptr;
  vector<ge::OpDescPtr> nullResult = {};
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(splitNodeName, "SplitV")),
      failStatus=true; return nullResult);
  ge::GeShape inputShape = GeShape({(matmulOutputDesc.GetOriginShape().GetDim(W_INDEX) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX});
  matmulOutputDesc.SetShape(inputShape);
  matmulOutputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  lstmSplitDesc->AddInputDesc("x", matmulOutputDesc);

  vector<int64_t> dx_dims;
  dx_dims.push_back((input_size + TANHCT_INDEX) / MASK_INDEX);
  dx_dims.push_back(batch_nz_size);
  dx_dims.push_back(MASK_INDEX);
  dx_dims.push_back(MASK_INDEX);

  vector<int64_t> dx_ori_dims;
  dx_ori_dims.push_back(batch_size);
  dx_ori_dims.push_back(input_size);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_ori_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y0", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back((hidden_size + TANHCT_INDEX) / MASK_INDEX);
  dh_dims.push_back(batch_nz_size);
  dh_dims.push_back(MASK_INDEX);
  dh_dims.push_back(MASK_INDEX);

  vector<int64_t> dh_ori_dims;
  dh_ori_dims.push_back(batch_size);
  dh_ori_dims.push_back(hidden_size);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  ge::GeShape dh_ori_shape(dh_ori_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y1", dh_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back((input_size + TANHCT_INDEX) / MASK_INDEX);
  size_splits.push_back((hidden_size + TANHCT_INDEX) / MASK_INDEX);

  ge::GeShape sizeSplitShape = ge::GeShape({2});
  auto sizeSplitDesc = ge::GeTensorDesc(sizeSplitShape, ge::FORMAT_ND, ge::DT_INT32);
  sizeSplitDesc.SetOriginShape(sizeSplitShape);
  sizeSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("size_splits", sizeSplitDesc);

  ge::GeTensorPtr sizeSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (sizeSplitDescTensor = std::make_shared<ge::GeTensor>(sizeSplitDesc)),
        failStatus=true; return nullResult);
  vector<int32_t> sizeSplitValue;
  sizeSplitValue.push_back(static_cast<int32_t>(input_size));
  sizeSplitValue.push_back(static_cast<int32_t>(hidden_size));

  sizeSplitDescTensor->SetData(reinterpret_cast<uint8_t *>(sizeSplitValue.data()),
                               sizeSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr sizeSplitOpDesc = ge::OpDescUtils::CreateConstOp(sizeSplitDescTensor);
  sizeSplitOpDesc->SetName(splitSizeNodeName);


  ge::GeShape sizeDimShape = ge::GeShape({W_INDEX});
  auto dimSplitDesc = ge::GeTensorDesc(sizeDimShape, ge::FORMAT_ND, ge::DT_INT32);
  dimSplitDesc.SetOriginShape(sizeDimShape);
  dimSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("split_dim", dimSplitDesc);

  ge::GeTensorPtr dimSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (dimSplitDescTensor = std::make_shared<ge::GeTensor>(dimSplitDesc)),
        failStatus=true; return nullResult);
  vector<int32_t> dimSplitValue;
  dimSplitValue.push_back(static_cast<int32_t>(W_INDEX));

  dimSplitDescTensor->SetData(reinterpret_cast<uint8_t *>(dimSplitValue.data()),
                              dimSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr dimSplitOpDesc = ge::OpDescUtils::CreateConstOp(dimSplitDescTensor);
  dimSplitOpDesc->SetName(splitDimNodeName);

  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);
  vector<string> depend_names = {"size_splits", "split_dim"};
  lstmSplitDesc->SetOpInferDepends(depend_names);
  vector<ge::OpDescPtr> result = {lstmSplitDesc, sizeSplitOpDesc, dimSplitOpDesc};

  return result;
}

ge::OpDescPtr GetDynamicBodyDxConcatNode(std::string cellNodeName, ge::NodePtr dynamicRNNGradNode,
                                         ge::ComputeGraph& graph, bool& failStatus,
                                         ge::GeTensorDesc splitInputDesc, ge::GeTensorDesc concatOriDesc)
{

  ge::OpDescPtr dxConcatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (dxConcatDesc = std::make_shared<ge::OpDesc>(cellNodeName, "ConcatD")),
       failStatus=true; return nullptr);
  dxConcatDesc->AddInputDesc("x0", splitInputDesc);
  dxConcatDesc->AddInputDesc("x1", concatOriDesc);

  dxConcatDesc->AddOutputDesc("y", concatOriDesc);

  ge::AttrUtils::SetInt(dxConcatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(dxConcatDesc, "N", 2);

  return dxConcatDesc;
}
ge::OpDescPtr DynamicRNNGradDFusionPass::GetDynamicDxConcatNode(std::string cellNodeName,
                                                                ge::NodePtr dynamicRNNGradNode,
                                                                ge::ComputeGraph& graph, bool& failStatus,
                                                                ge::GeTensorDesc splitInputDesc,
                                                                ge::GeTensorDesc concatOriDesc)
{
  ge::OpDescPtr dxConcatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (dxConcatDesc = std::make_shared<ge::OpDesc>(cellNodeName, "ConcatD")),
      failStatus=true; return nullptr);
  dxConcatDesc->AddInputDesc("x0", splitInputDesc);
  dxConcatDesc->AddInputDesc("x1", concatOriDesc);

  dxConcatDesc->AddOutputDesc("y", concatOriDesc);

  ge::AttrUtils::SetInt(dxConcatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(dxConcatDesc, "N", 2);

  return dxConcatDesc;
}

ge::ComputeGraphPtr DynamicRNNGradDFusionPass::BuildBodyGraph(ge::ComputeGraph &graph, ge::NodePtr &whileNode,
                                                              int32_t argNum, ge::NodePtr dynamicRNNGradNode,
                                                              ge::GeTensorDesc concatOriDesc,
                                                              ge::GeTensorDesc concatDgateOriDesc, bool& failStatus)
{
  std::string body_name = DynamicRNNGradName + "Body";
  CompleteGraphBuilder graph_builder(body_name, false);
  graph_builder.SetParentNode(whileNode);
  std::string cellNodeName = DynamicRNNGradName + "DynamicLSTMGradCell";
  ge::OpDescPtr cellNode = GetDynamicLSTMGradCellNode(cellNodeName, dynamicRNNGradNode,
                                    whileNode->GetOpDesc()->GetInputDesc(O_INDEX).Clone(), graph, failStatus);
  graph_builder.AddNode(cellNode);
  std::string matmulNodeName = DynamicRNNGradName + "MatMulCell";
  ge::OpDescPtr matmulNode = GetDynamicMatMulNode(matmulNodeName, dynamicRNNGradNode,
                                               graph, failStatus, cellNode->GetOutputDesc(X_INDEX).GetShape());
  graph_builder.AddNode(matmulNode);
  std::string splitNodeName = DynamicRNNGradName + "SplitCell";
  std::string splitDimNodeName = DynamicRNNGradName + "SplitDimConst";
  std::string splitSizeNodeName = DynamicRNNGradName + "SplitSizeConst";
  vector<ge::OpDescPtr> result = GetDynamicBodySplitNode(splitNodeName, splitDimNodeName, splitSizeNodeName,
                              dynamicRNNGradNode, graph, failStatus, matmulNode->GetOutputDesc(X_INDEX).Clone());
  graph_builder.AddNode(result[X_INDEX]);
  graph_builder.AddNode(result[W_INDEX]);
  graph_builder.AddNode(result[2]);
  std::string concatDxNodeName = DynamicRNNGradName + "dxConcatCell";
  ge::GeTensorDesc splitInputDesc = result[X_INDEX]->GetOutputDesc(X_INDEX).Clone();
  ge::OpDescPtr concatDxDesc = GetDynamicDxConcatNode(concatDxNodeName, dynamicRNNGradNode, graph,
                                                      failStatus, splitInputDesc, concatOriDesc);
  graph_builder.AddNode(concatDxDesc);
  string bodyDxReshapeNodeName = DynamicRNNGradName + "bodyDxReshapeNode";
  string constDxReshapeInputName = DynamicRNNGradName + "dxReshapeConst";
  vector<ge::OpDescPtr> bodyDxReshapeNodes = GetDynamicBodyDxReshapeNode(bodyDxReshapeNodeName, constDxReshapeInputName,
                                              dynamicRNNGradNode, result[X_INDEX]->GetOutputDesc(X_INDEX).Clone(),
                                              concatDxDesc->GetInputDesc(X_INDEX).Clone(), graph, failStatus);
  graph_builder.AddNode(bodyDxReshapeNodes[X_INDEX]);
  std::string dgateConcatNodeName = DynamicRNNGradName + "dgateConcatCell";
  ge::GeTensorDesc dgateInputDesc = cellNode->GetOutputDesc(X_INDEX).Clone();
  vector<int64_t> dgateShape = {W_INDEX, dgateInputDesc.GetShape().GetDim(X_INDEX), dgateInputDesc.GetShape().GetDim(W_INDEX)};
  dgateInputDesc.SetShape(GeShape(dgateShape));
  dgateInputDesc.SetOriginShape(GeShape(dgateShape));
  ge::OpDescPtr concatDgateDesc = GetDynamicBodyDxConcatNode(dgateConcatNodeName, dynamicRNNGradNode, graph,
                                                             failStatus, dgateInputDesc, concatDgateOriDesc);
  graph_builder.AddNode(concatDgateDesc);

  string bodyReshapeNodeName = DynamicRNNGradName + "bodyDgateReshapeNode";
  string constReshapeInputName = DynamicRNNGradName + "reshapeConst";
  vector<ge::OpDescPtr> bodyDgateReshapeNodes = GetDynamicBodyReshapeNode(bodyReshapeNodeName, constReshapeInputName,
                                                                          dynamicRNNGradNode,
                                                                          cellNode->GetOutputDesc(X_INDEX).Clone(),
                                                                          concatDgateDesc->GetInputDesc(X_INDEX).Clone(),
                                                                          graph, failStatus);
  graph_builder.AddNode(bodyDgateReshapeNodes[X_INDEX]);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  uint32_t idxOri = W_INDEX;
  int64_t idxDgate = X_INDEX;

  if (direction == "UNIDIRECTIONAL") {
    idxOri = W_INDEX;
    idxDgate = X_INDEX;
  } else {
    idxOri = X_INDEX;
    idxDgate = W_INDEX;
  }

  string constName = DynamicRNNGradName + "OneConst";
  graph_builder.AddNode(CreateConstDesc(constName, W_INDEX, "int32"));
  std::string addName = DynamicRNNGradName + "Add";
  OpDescBuilder op_desc_builder(addName, "Add");
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(O_INDEX).Clone())
                 .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(O_INDEX).Clone())
                 .AddOutput("y", whileNode->GetOpDesc()->GetInputDesc(O_INDEX).Clone());
  graph_builder.AddNode(op_desc_builder.Build());
  graph_builder.SetInput(2, {cellNodeName}, {X_INDEX})
               .SetInput(3, {cellNodeName}, {W_INDEX})
               .SetInput(INIT_H_INDEX, {cellNodeName}, {2})
               .SetInput(INIT_C_INDEX, {cellNodeName}, {3})
               .SetInput(H_INDEX, {cellNodeName}, {INIT_H_INDEX})
               .SetInput(C_INDEX, {cellNodeName}, {INIT_C_INDEX})
               .SetInput(DY_INDEX, {cellNodeName}, {H_INDEX})
               .SetInput(DH_INDEX, {cellNodeName}, {C_INDEX})
               .SetInput(DC_INDEX, {cellNodeName}, {DY_INDEX})
               .SetInput(I_INDEX, {cellNodeName}, {DH_INDEX})
               .SetInput(J_INDEX, {cellNodeName}, {I_INDEX})
               .SetInput(O_INDEX, {cellNodeName, addName}, {DC_INDEX, X_INDEX});
  graph_builder.SetUselessInput(TANHCT_INDEX);
  graph_builder.SetInput(W_INDEX, {matmulNodeName}, {W_INDEX});
  graph_builder.SetInput(X_INDEX, {concatDxNodeName}, {W_INDEX});
  graph_builder.SetInput(F_INDEX, {dgateConcatNodeName}, {idxOri});

  graph_builder.SetInput(MASK_INDEX, {bodyDxReshapeNodeName}, {W_INDEX});
  graph_builder.SetInput(17, {bodyReshapeNodeName}, {W_INDEX});

  graph_builder.AddDataLink(cellNodeName, X_INDEX, matmulNodeName, X_INDEX)
               .AddDataLink(matmulNodeName, X_INDEX, splitNodeName, X_INDEX)
               .AddDataLink(splitNodeName, X_INDEX, bodyDxReshapeNodeName, X_INDEX)
               .AddDataLink(bodyDxReshapeNodeName, X_INDEX, concatDxNodeName, X_INDEX)
               .AddDataLink(cellNodeName, X_INDEX, bodyReshapeNodeName, X_INDEX)
               .AddDataLink(bodyReshapeNodeName, X_INDEX, dgateConcatNodeName, idxDgate)
               .AddDataLink(splitDimNodeName, X_INDEX, splitNodeName, 2)
               .AddDataLink(splitSizeNodeName, X_INDEX, splitNodeName, W_INDEX)
               .AddDataLink(constName, X_INDEX, addName, W_INDEX);
  graph_builder.AddOutput(concatDxNodeName, X_INDEX);

  for (uint32_t i = W_INDEX; i < INIT_C_INDEX; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), X_INDEX);
  }
  graph_builder.AddOutput(splitNodeName, W_INDEX);
  graph_builder.AddOutput(cellNodeName, W_INDEX);
  for (uint32_t i = C_INDEX; i < I_INDEX; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), X_INDEX);
  }
  graph_builder.AddOutput("Data_" + std::to_string(I_INDEX), X_INDEX);
  graph_builder.AddOutput("Data_" + std::to_string(J_INDEX), X_INDEX);
  graph_builder.AddOutput(dgateConcatNodeName, X_INDEX);
  graph_builder.AddOutput(addName, X_INDEX);
  graph_builder.AddOutput("Data_" + std::to_string(TANHCT_INDEX), X_INDEX);

  graph_builder.AddOutput("Data_" + std::to_string(MASK_INDEX), X_INDEX);
  graph_builder.AddOutput("Data_" + std::to_string(17), X_INDEX);
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = X_INDEX; i < argNum; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  std::map<uint32_t, uint32_t> output_mapping;
  for (int32_t i = X_INDEX; i < argNum; i++) {
    output_mapping[i] = i;
  }
  graph_builder.SetOutputMapping(output_mapping);
  ge::graphStatus error_code = ge::GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr body_graph = graph_builder.Build(error_code, error_msg);

  if (body_graph == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Build body_graph failed: error_code:%u.", error_code);
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Build body_graph failed: error_msg:%s.", error_msg.c_str());
    return nullptr;
  }

  size_t index = whileNode->GetOpDesc()->GetSubgraphInstanceNames().size();
  whileNode->GetOpDesc()->AddSubgraphName(DynamicRNNGradName +"Body");
  whileNode->GetOpDesc()->SetSubgraphInstanceName(index, body_name);

  return body_graph;
}

vector<ge::NodePtr> DynamicRNNGradDFusionPass::BuildWhileNodes(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                               vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                               ge::GeTensorDesc concatOriDesc,
                                                               ge::GeTensorDesc concatDgateOriDesc,
                                                               ge::GeTensorDesc curTDesc, ge::GeTensorDesc tDesc
                                                               , ge::GeTensorDesc reshapeDxDesc, ge::GeTensorDesc
                                                               reshapeDgateDesc)
{
  OpDescBuilder op_desc_builder(DynamicRNNGradName + "While_Op", "While");
  OpDescPtr op_desc = op_desc_builder
        .AddInput("input0", concatOriDesc)
        .AddInput("input1", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["w"]).Clone())
        .AddInput("input2", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).Clone())
        .AddInput("input3", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["c"]).Clone())
        .AddInput("input4", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dy"]).Clone())
        .AddInput("input5", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).Clone())
        .AddInput("input6", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).Clone())
        .AddInput("input7", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]).Clone())
        .AddInput("input8", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["j"]).Clone())
        .AddInput("input9", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["f"]).Clone())
        .AddInput("input10", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["o"]).Clone())
        .AddInput("input11",
                    dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["tanhct"]).Clone())
        .AddInput("input12", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]).Clone())
        .AddInput("input13", concatDgateOriDesc)
        .AddInput("input14", curTDesc)
        .AddInput("input15", tDesc)
        .AddInput("input16", reshapeDxDesc)
        .AddInput("input17", reshapeDgateDesc)
        .AddOutput("output0", concatOriDesc)
        .AddOutput("output1", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["w"]).Clone())
        .AddOutput("output2",
                  dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).Clone())
        .AddOutput("output3", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["c"]).Clone())
        .AddOutput("output4", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dy"]).Clone())
        .AddOutput("output5", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).Clone())
        .AddOutput("output6", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).Clone())
        .AddOutput("output7", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]).Clone())
        .AddOutput("output8", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["j"]).Clone())
        .AddOutput("output9", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["f"]).Clone())
        .AddOutput("output10", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["o"]).Clone())
        .AddOutput("output11",
                  dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["tanhct"]).Clone())
        .AddOutput("output12", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]).Clone())
        .AddOutput("output13", concatDgateOriDesc)
        .AddOutput("output14", curTDesc)
        .AddOutput("output15", tDesc)
        .AddOutput("output16", reshapeDxDesc)
        .AddOutput("output17", reshapeDgateDesc).Build();
  if (op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create while opdesc failed, name:WHILE");
    return {};
  }
  ge::NodePtr whileNode = graph.AddNode(op_desc);
  if (whileNode == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create while node failed, name:WHILE");
    return {};
  }

  ge::ComputeGraphPtr cond_graph = BuildCondGraph(whileNode, 18);
  if ((cond_graph == nullptr) || (graph.AddSubgraph(cond_graph) != ge::GRAPH_SUCCESS)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add while_cond_graph failed.");
    return {};
  }
  ge::ComputeGraphPtr body_graph = BuildBodyGraph(graph, whileNode, 18, dynamicRNNGradNode, concatOriDesc,
                                                  concatDgateOriDesc, failStatus);
  if ((body_graph == nullptr) || (graph.AddSubgraph(body_graph) != ge::GRAPH_SUCCESS)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add while_body_graph failed.");
    return {};
  }
  auto graphNodes = cond_graph->GetAllNodes();
  auto bodyNodes = body_graph->GetAllNodes();
  vector<ge::NodePtr> result = {};
  for (const NodePtr &node: graphNodes) {
    result.push_back(node);
  }
  for (const NodePtr &node: bodyNodes) {
    result.push_back(node);
  }
  result.push_back(whileNode);
  return result;
}

vector<ge::NodePtr> DynamicRNNGradDFusionPass::BuildT0Graph(ge::NodePtr dynamicRNNGradNode,
                                                            ge::GeTensorDesc curTDesc, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus)
{
  std::string cellNodeName = DynamicRNNGradName + "DynamicLSTMGradCell0";
  ge::OpDescPtr cellNode = GetDynamicLSTMGradCellNode(cellNodeName, dynamicRNNGradNode, curTDesc, graph, failStatus);
  ge::NodePtr t0CellNode = graph.AddNode(cellNode);
  std::string t0MatmulNodeName = DynamicRNNGradName + "MatMulCell0";

  ge::OpDescPtr matmulNode = GetDynamicMatMulNode(t0MatmulNodeName, dynamicRNNGradNode, graph, failStatus,
                                                  cellNode->GetOutputDesc(X_INDEX).GetShape());
  ge::NodePtr t0MatmulNode = graph.AddNode(matmulNode);

  std::string splitNodeName = DynamicRNNGradName + "SplitCell0";
  std::string splitDimNodeName = DynamicRNNGradName + "SplitDimConst0";
  std::string splitSizeNodeName = DynamicRNNGradName + "SplitSizeConst0";

  vector<ge::OpDescPtr> result = GetDynamicSplitNode(splitNodeName, splitDimNodeName, splitSizeNodeName,
                                                     dynamicRNNGradNode, graph, failStatus,
                                                     matmulNode->GetOutputDesc(X_INDEX).Clone());
  ge::NodePtr t0SplitNode = graph.AddNode(result[X_INDEX]);
  ge::NodePtr t0SizeSplitNode = graph.AddNode(result[W_INDEX]);
  ge::NodePtr t0DimSplitNode = graph.AddNode(result[2]);

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).
                                                           GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeInitC->GetOutDataAnchor(X_INDEX),
                            t0CellNode->GetInDataAnchor(X_INDEX));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INIT_C_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(X_INDEX));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(C_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(DY_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(2));
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeDh->GetOutDataAnchor(X_INDEX), t0CellNode->GetInDataAnchor(3));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(DH_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(3));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeDc->GetOutDataAnchor(X_INDEX), t0CellNode->GetInDataAnchor(INIT_H_INDEX));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(DC_INDEX)->GetPeerOutAnchor(),
                 t0CellNode->GetInDataAnchor(INIT_H_INDEX));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(I_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(INIT_C_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(J_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(H_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(F_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(C_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(O_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(DY_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(TANHCT_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(DH_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(MASK_INDEX)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(I_INDEX));

  string constName = DynamicRNNGradName + "curT0Const";
  ge::OpDescPtr curTConst = CreateConstDesc(constName, X_INDEX, "int32");
  ge::NodePtr curTConstNode = graph.AddNode(curTConst);
  FUSION_PASS_CHECK(curTConstNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  t0CellNode->AddLinkFrom(DC_INDEX, curTConstNode);

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(W_INDEX)->GetPeerOutAnchor(), t0MatmulNode->GetInDataAnchor(W_INDEX));

  ge::GraphUtils::AddEdge(t0CellNode->GetOutDataAnchor(X_INDEX), t0MatmulNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(t0MatmulNode->GetOutDataAnchor(X_INDEX), t0SplitNode->GetInDataAnchor(X_INDEX));

  t0SplitNode->AddLinkFrom(W_INDEX, t0SizeSplitNode);
  t0SplitNode->AddLinkFrom(2, t0DimSplitNode);

  return {t0CellNode, t0SplitNode};
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                                            ge::NodePtr &while_node, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul",
                                                 "BatchMatMul")),
      failStatus=true; return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescXh = ge::GeTensorDesc(concatNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetOriginShape(),
                                                        ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescXh.SetOriginShape(concatNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetOriginShape());
  inputTensorDescXh.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorDesc inputTensorDescXhTotal = inputTensorDescXh.Clone();
  vector<int64_t> x1NzDims = {inputTensorDescXhTotal.GetOriginShape().GetDim(X_INDEX),
                              (inputTensorDescXhTotal.GetOriginShape().GetDim(2) + TANHCT_INDEX) / MASK_INDEX, batch_nz_size, MASK_INDEX, MASK_INDEX};
  inputTensorDescXhTotal.SetShape(GeShape(x1NzDims));
  inputTensorDescXhTotal.SetFormat(ge::FORMAT_FRACTAL_NZ);

  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(W_INDEX, 1000));
  x1_range.push_back(std::make_pair((hidden_size + input_size) / MASK_INDEX, (hidden_size + input_size) / MASK_INDEX));
  x1_range.push_back(std::make_pair(batch_start, batch_end));
  x1_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  x1_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  inputTensorDescXhTotal.SetShapeRange(x1_range);

  vector<int64_t> x2NzDims = {while_node->GetOpDesc()->GetOutputDesc(F_INDEX).GetOriginShape().GetDim(X_INDEX),
                             (while_node->GetOpDesc()->GetOutputDesc(F_INDEX).GetOriginShape().GetDim(2) + TANHCT_INDEX) / MASK_INDEX,
                             batch_nz_size, MASK_INDEX, MASK_INDEX};
  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(GeShape(x2NzDims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(while_node->GetOpDesc()->GetOutputDesc(F_INDEX).GetOriginShape());
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::make_pair(W_INDEX, 1000));
  x2_range.push_back(std::make_pair((hidden_size * INIT_H_INDEX) / MASK_INDEX, (hidden_size * INIT_H_INDEX) / MASK_INDEX));
  x2_range.push_back(std::make_pair(batch_start, batch_end));
  x2_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  x2_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  inputTensorDescDgate.SetShapeRange(x2_range);

  matmulDesc->AddInputDesc("x1", inputTensorDescXhTotal);
  matmulDesc->AddInputDesc("x2", inputTensorDescDgate);

  vector<int64_t> outputDims;
  outputDims.push_back(t_size);
  outputDims.push_back(input_size + hidden_size);
  outputDims.push_back(hidden_size * INIT_H_INDEX);

  vector<int64_t> outputNzDims;
  outputNzDims.push_back(t_size);
  outputNzDims.push_back((hidden_size * INIT_H_INDEX + TANHCT_INDEX) / MASK_INDEX);
  outputNzDims.push_back((input_size + hidden_size + TANHCT_INDEX) / MASK_INDEX);
  outputNzDims.push_back(MASK_INDEX);
  outputNzDims.push_back(MASK_INDEX);
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputNzDims);

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_range.push_back(std::make_pair(W_INDEX, 1000));
  y_range.push_back(std::make_pair((hidden_size + input_size) / MASK_INDEX, (hidden_size + input_size) / MASK_INDEX));
  y_range.push_back(std::make_pair((hidden_size * INIT_H_INDEX) / MASK_INDEX, (hidden_size * INIT_H_INDEX) / MASK_INDEX));
  y_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  y_range.push_back(std::make_pair(MASK_INDEX, MASK_INDEX));
  outputTensorDesc.SetShapeRange(y_range);

  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   matmulNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(matmulNode);
  // Edge
  ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(X_INDEX), matmulNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(while_node->GetOutDataAnchor(F_INDEX), matmulNode->GetInDataAnchor(W_INDEX));  // dgate

  return matmulNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode,
                                                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                            bool& failStatus){
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
                    (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() +
                                                               "LSTMWeightGrad/Dw/ConcatD", "ConcatD")),
                    failStatus = true; return nullptr);

  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(X_INDEX);
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();

  concatDesc->AddInputDesc("x0", inputTensorDescX);
  concatDesc->AddInputDesc("x1", inputTensorDescH);

  vector<int64_t> outputDims;
  outputDims.push_back(t_size);
  outputDims.push_back(batch_size);
  outputDims.push_back(input_size + hidden_size);
  ge::GeShape outputShape(outputDims);

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  concatDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  newNodes.push_back(concatNode);

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(), concatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(X_INDEX), concatNode->GetInDataAnchor(W_INDEX));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                                             ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                             bool& failStatus){
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
                    (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() +
                                                               "LSTMWeightGrad/Dw/HConcatD",
                                                               "ConcatD")),
                    failStatus = true; return nullptr);
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INIT_H_INDEX);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();

  vector<int64_t> input_h;
  input_h.push_back(W_INDEX);
  input_h.push_back(batch_size);
  input_h.push_back(hidden_size);
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  int64_t idxInitH = X_INDEX;
  int64_t idxSplit = W_INDEX;
  if (direction == "UNIDIRECTIONAL") {
    concatDesc->AddInputDesc("x0", inputTensorDescInitH);
    concatDesc->AddInputDesc("x1", inputTensorDescSplitH);
  } else {
    concatDesc->AddInputDesc("x0", inputTensorDescSplitH);
    concatDesc->AddInputDesc("x1", inputTensorDescInitH);
    idxInitH = W_INDEX;
    idxSplit = X_INDEX;
  }

  vector<int64_t> outputDims = {};
  outputDims.push_back(t_size);
  outputDims.push_back(batch_size);
  outputDims.push_back(hidden_size);
  ge::GeShape outputShape(outputDims);

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescInitH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  concatDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", X_INDEX);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  newNodes.push_back(concatNode);

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_h"]).GetShape().GetDims().size() ==
      2) {
    std::string reshapeHName = DynamicRNNGradName + "_initHReshapeNode";
    reshapeInitH = DynamicAddInithReshapeNode(
        dynamicRNNGradNode, reshapeHName,
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_h"]).Clone(), graph, newNodes,
        failStatus);
    ge::GraphUtils::AddEdge(
        dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
        reshapeInitH->GetInDataAnchor(X_INDEX));
    ge::GraphUtils::AddEdge(reshapeInitH->GetOutDataAnchor(X_INDEX), concatNode->GetInDataAnchor(idxInitH));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INIT_H_INDEX)->GetPeerOutAnchor(),
                            concatNode->GetInDataAnchor(idxInitH));
  }

  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(X_INDEX), concatNode->GetInDataAnchor(idxSplit));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr& sizeConcatNode,
                                                           ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus){
  ge::OpDescPtr sliceDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (sliceDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "RNNWeightGrad/Dh/Slice", "Slice")),
        failStatus=true; return nullptr);
  ge::GeTensorDesc sliceInputDesc =
                      dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["h"]).Clone();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(W_INDEX, 1000));
  x1_range.push_back(std::make_pair(W_INDEX, 1000));
  x1_range.push_back(std::make_pair(hidden_size, hidden_size));
  sliceInputDesc.SetShapeRange(x1_range);
  sliceDesc->AddInputDesc("x", sliceInputDesc);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  vector<int64_t> output1Dim = {X_INDEX, X_INDEX, X_INDEX};
  if (direction != "UNIDIRECTIONAL") {
    output1Dim = {W_INDEX, X_INDEX, X_INDEX};
  }
  GeTensorDesc offsetDesc(GeShape({static_cast<int64_t>(output1Dim.size())}), FORMAT_ND, DT_INT64);
  sliceDesc->AddInputDesc("offsets", offsetDesc);
  ge::OpDescPtr offsetConst = CreateListConstDesc("addWhileHSliceOffset", output1Dim);
  ge::NodePtr offsetNode = graph.AddNode(offsetConst);
  FUSION_PASS_CHECK(offsetNode == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  newNodes.push_back(offsetNode);
  vector<int64_t> output2Dim = {-W_INDEX, batch_size, hidden_size};

  GeTensorDesc sizeDesc(GeShape({static_cast<int64_t>(output2Dim.size())}), FORMAT_ND, DT_INT64);
  sliceDesc->AddInputDesc("size", sizeDesc);

  ge::GeTensorDesc outDesc = dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(RNN_GRAD_NODE_INPUT_INDEX["h"]).Clone();
  outDesc.SetShape(GeShape(output2Dim));
  outDesc.SetOriginShape(GeShape(output2Dim));
  outDesc.SetFormat(FORMAT_ND);
  sliceDesc->AddOutputDesc("y", outDesc);

  vector<string> depend_names = {"offsets", "size"};
  sliceDesc->SetOpInferDepends(depend_names);
  ge::NodePtr sliceNode = graph.AddNode(sliceDesc);
  newNodes.push_back(sliceNode);

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["h"])->GetPeerOutAnchor(),
                                                              sliceNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(offsetNode->GetOutDataAnchor(X_INDEX), sliceNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(sizeConcatNode->GetOutDataAnchor(X_INDEX), sliceNode->GetInDataAnchor(2));

  return sliceNode;

}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddConcatHCNode(ge::NodePtr dynamicRNNGradNode,
                                                              ge::NodePtr& sizeConcatNode, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus){
  ge::NodePtr splitNode = DynamicAddSplitNode(dynamicRNNGradNode, sizeConcatNode, graph, newNodes, failStatus);

  ge::NodePtr hConcatNode = DynamicAddHConcatNode(dynamicRNNGradNode, splitNode, graph, newNodes, failStatus);

  ge::NodePtr concatNode = DynamicAddConcatNode(dynamicRNNGradNode, hConcatNode, graph, newNodes, failStatus);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddInputReshapeNode(ge::NodePtr dynamicRNNGradNode, string reshapeName,
                                                              ge::GeTensorDesc inputDesc, ge::ComputeGraph& graph,
                                                              vector<ge::NodePtr>& newNodes, bool& failStatus){
  std::string operatorName = dynamicRNNGradNode->GetName() + "/" + reshapeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  vector<int64_t> outputReshapeDims = {inputDesc.GetShape().GetDim(W_INDEX), inputDesc.GetShape().GetDim(2)};
  ge::GeShape outputReshapeShape(outputReshapeDims);

  ge::GeTensorDesc reshapeCellOutputDesc = ge::GeTensorDesc(outputReshapeShape, ge::FORMAT_ND, inputDesc.GetDataType());
  reshapeCellOutputDesc.SetOriginShape(outputReshapeShape);
  reshapeCellOutputDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc), OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                         "Reshape node update outputDesc failed!"),
                    return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc),
                                                              OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                                         "Reshape node update outputDesc failed!"),
                    return nullptr);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  FUSION_PASS_CHECK(myReshape_node==nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Create node error"), return nullptr);

  return myReshape_node;
}

ge::NodePtr DynamicRNNGradDFusionPass::DynamicAddInithReshapeNode(ge::NodePtr dynamicRNNGradNode, string reshapeName,
                                                                  ge::GeTensorDesc inputDesc, ge::ComputeGraph& graph,
                                                                  vector<ge::NodePtr>& newNodes, bool& failStatus) {
  std::string operatorName = dynamicRNNGradNode->GetName() + "/" + reshapeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "Unsqueeze");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  vector<int64_t> outputReshapeDims = {W_INDEX, inputDesc.GetShape().GetDim(X_INDEX), inputDesc.GetShape().GetDim(W_INDEX)};
  ge::GeShape outputReshapeShape(outputReshapeDims);

  ge::GeTensorDesc reshapeCellOutputDesc = ge::GeTensorDesc(outputReshapeShape, ge::FORMAT_ND, inputDesc.GetDataType());
  reshapeCellOutputDesc.SetOriginShape(outputReshapeShape);
  reshapeCellOutputDesc.SetOriginFormat(ge::FORMAT_ND);

  // shape range
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_size, hidden_size));
  x1_range.insert(x1_range.begin(), std::make_pair(W_INDEX, -W_INDEX));
  inputDesc.SetShapeRange(x1_range);
  inputDesc.SetOriginShapeRange(x1_range);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);

  // set attr
  ge::AttrUtils::SetListInt(reshape_desc, "axes", {X_INDEX});

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  FUSION_PASS_CHECK(myReshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Create node error"), return nullptr);

  return myReshape_node;
}

vector<ge::NodePtr> DynamicRNNGradDFusionPass::DynamicAddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode,
                                                                           ge::ComputeGraph& graph,
                                                                           vector<ge::NodePtr>& newNodes,
                                                                           bool& failStatus){
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::OpDescPtr currTConst = CreateConstDesc(DynamicRNNGradName + "currT", W_INDEX, "int32");
  ge::NodePtr currTNode = graph.AddNode(currTConst);
  FUSION_PASS_CHECK(currTNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  newNodes.push_back(currTNode);
  ge::NodePtr shapeNode = BuildTShape(dynamicRNNGradDesc->GetInputDesc(X_INDEX).Clone(),
                                      dynamicRNNGradNode, graph, failStatus);
  ge::NodePtr tSplitNode = BuildTSplit(shapeNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone(),
                                       dynamicRNNGradNode, graph, failStatus);
  vector<ge::NodePtr> t0Nodes = BuildT0Graph(dynamicRNNGradNode, currTConst->GetOutputDesc(X_INDEX).Clone(),
                                             graph, newNodes, failStatus);

  ge::NodePtr cellT0Node = t0Nodes[X_INDEX];
  ge::NodePtr splitT0Node = t0Nodes[W_INDEX];

  string dxReshapeNodeName = DynamicRNNGradName + "dxReshapeNode";
  ge::NodePtr dxReshapeNode = GetDynamicReshapeDxNode(dxReshapeNodeName, dynamicRNNGradNode,
                                                      splitT0Node->GetOpDesc()->GetOutputDesc(X_INDEX).Clone(),
                                                      cellT0Node->GetOpDesc()->GetOutputDesc(X_INDEX).Clone(),
                                                      graph, failStatus);
  ge::GeShape shapeShape = ge::GeShape({W_INDEX,});
  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorPtr shapeDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (shapeDescTensor = std::make_shared<ge::GeTensor>(shapeDesc)),
        failStatus=true; return {});
  vector<int64_t> shapeValue;
  int64_t dim1 = -W_INDEX;
  shapeValue.push_back(static_cast<int64_t>(dim1));

  shapeDescTensor->SetData(reinterpret_cast<uint8_t *>(shapeValue.data()), shapeValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(shapeDescTensor);
  ge::NodePtr dxreshape_const_node = graph.AddNode(const_opdesc);
  FUSION_PASS_CHECK(dxreshape_const_node == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  std::string reshapeDxNodeName = "TDxConcat";
  ge::NodePtr dxReshapeConcatNode = BuildDxReshapeSizeConcatNode(dynamicRNNGradNode, reshapeDxNodeName,
                                                                 dxreshape_const_node, graph, failStatus);
  newNodes.push_back(dxReshapeConcatNode);
  ge::GeTensorDesc concatOriDesc = t0Nodes[W_INDEX]->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  concatOriDesc.SetShape(concatOriDesc.GetOriginShape());
  concatOriDesc.SetFormat(ge::FORMAT_ND);
  vector<int64_t> dxConcatDims = {-W_INDEX, batch_size, concatOriDesc.GetOriginShape().GetDim(W_INDEX)};

  concatOriDesc.SetShape(GeShape(dxConcatDims));
  concatOriDesc.SetOriginShape(GeShape(dxConcatDims));

  ge::GeTensorDesc concatDgateOriDesc = t0Nodes[X_INDEX]->GetOpDesc()->GetOutputDesc(X_INDEX).Clone();
  vector<int64_t> dgateConcatDims = {-W_INDEX, batch_size, concatDgateOriDesc.GetShape().GetDim(W_INDEX)};
  concatDgateOriDesc.SetShape(GeShape(dgateConcatDims));
  concatDgateOriDesc.SetOriginShape(GeShape(dgateConcatDims));

  ge::OpDescPtr totalTConst = CreateConstDesc(DynamicRNNGradName + "totalT", 2, "int32");
  ge::NodePtr totalTNode = graph.AddNode(totalTConst);
  FUSION_PASS_CHECK(totalTNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  newNodes.push_back(totalTNode);
  ge::GeShape shapeBodyShape = ge::GeShape({W_INDEX,});
  auto shapeBodyDesc = ge::GeTensorDesc(shapeBodyShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeBodyDesc.SetOriginShape(shapeBodyShape);
  shapeBodyDesc.SetOriginFormat(ge::FORMAT_ND);
  ge::GeTensorPtr shapeBodyDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
       (shapeBodyDescTensor = std::make_shared<ge::GeTensor>(shapeBodyDesc)),
        failStatus=true; return {});
  vector<int64_t> shapeBodyValue;
  shapeBodyValue.push_back(static_cast<int64_t>(W_INDEX));

  shapeBodyDescTensor->SetData(reinterpret_cast<uint8_t *>(shapeBodyValue.data()),
                                               shapeBodyValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_body_opdesc = ge::OpDescUtils::CreateConstOp(shapeBodyDescTensor);
  ge::NodePtr dxreshape_body_const_node = graph.AddNode(const_body_opdesc);
  FUSION_PASS_CHECK(dxreshape_body_const_node == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  std::string reshapeBodyDxNodeName = "TDxBodyConcat";
  ge::NodePtr dxBodyReshapeConcatNode = BuildDxReshapeSizeConcatNode(dynamicRNNGradNode, reshapeBodyDxNodeName,
                                                                     dxreshape_body_const_node, graph, failStatus);
  newNodes.push_back(dxBodyReshapeConcatNode);
  vector<ge::NodePtr> whileNodes = BuildWhileNodes(dynamicRNNGradNode, graph, newNodes, failStatus, concatOriDesc,
                                                   concatDgateOriDesc, currTConst->GetOutputDesc(X_INDEX).Clone(),
                                                   totalTConst->GetOutputDesc(X_INDEX).Clone()
                                                   , dxBodyReshapeConcatNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone(),
                                                   dxBodyReshapeConcatNode->GetOpDesc()->GetOutputDesc(X_INDEX).Clone());

  int whileNodeSize = whileNodes.size();
  for (int i = X_INDEX; i < whileNodeSize; i++) {
    newNodes.push_back(whileNodes[i]);
  }

  string reshapeNodeName = "dgateReshapeNode";
  vector<ge::NodePtr> dgateReshapeResult = GetDynamicReshapeNode(reshapeNodeName, dynamicRNNGradNode, cellT0Node,
                                                                 whileNodes[X_INDEX]->GetOpDesc()->GetOutputDesc(F_INDEX).Clone(),
                                                                 shapeNode, graph, failStatus);
  ge::NodePtr dgateReshapeNode = dgateReshapeResult[X_INDEX];
  ge::NodePtr dgateBodyReshapeConcatNode = dgateReshapeResult[W_INDEX];
  ge::NodePtr whileNode = whileNodes[whileNodeSize - W_INDEX];
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(X_INDEX)->GetPeerOutAnchor(), shapeNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(shapeNode->GetOutDataAnchor(X_INDEX), tSplitNode->GetInDataAnchor(X_INDEX));

  ge::GraphUtils::AddEdge(dxreshape_const_node->GetOutDataAnchor(X_INDEX), dxReshapeConcatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(W_INDEX), dxReshapeConcatNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(dxReshapeConcatNode->GetOutDataAnchor(X_INDEX), dxReshapeNode->GetInDataAnchor(W_INDEX));

  ge::GraphUtils::AddEdge(splitT0Node->GetOutDataAnchor(X_INDEX), dxReshapeNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(dxReshapeNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(X_INDEX));

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(W_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(W_INDEX));

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).
                                                     GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeInitC->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(2));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INIT_C_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(2));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(C_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(DY_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(INIT_H_INDEX));

  ge::GraphUtils::AddEdge(splitT0Node->GetOutDataAnchor(W_INDEX), whileNode->GetInDataAnchor(INIT_C_INDEX));
  ge::GraphUtils::AddEdge(cellT0Node->GetOutDataAnchor(W_INDEX), whileNode->GetInDataAnchor(H_INDEX));

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(I_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(C_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(J_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(DY_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(F_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(DH_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(O_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(DC_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(TANHCT_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(I_INDEX));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(MASK_INDEX)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(J_INDEX));
  ge::GraphUtils::AddEdge(cellT0Node->GetOutDataAnchor(X_INDEX), dgateReshapeNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(dgateReshapeNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(F_INDEX));
  ge::GraphUtils::AddEdge(currTNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(O_INDEX));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(TANHCT_INDEX));

  ge::GraphUtils::AddEdge(dxreshape_body_const_node->GetOutDataAnchor(X_INDEX), dxBodyReshapeConcatNode->GetInDataAnchor(X_INDEX));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(W_INDEX), dxBodyReshapeConcatNode->GetInDataAnchor(W_INDEX));
  ge::GraphUtils::AddEdge(dxBodyReshapeConcatNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(MASK_INDEX));
  ge::GraphUtils::AddEdge(dgateBodyReshapeConcatNode->GetOutDataAnchor(X_INDEX), whileNode->GetInDataAnchor(17));
  if (dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > X_INDEX) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(X_INDEX), inAnchorPtr);
    }
  }

  if (dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() > X_INDEX) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(INIT_C_INDEX), inAnchorPtr);
    }
  }

  if (dynamicRNNGradNode->GetOutDataAnchor(INIT_H_INDEX)->GetPeerInDataAnchors().size() > X_INDEX) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(INIT_H_INDEX)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(H_INDEX), inAnchorPtr);
    }
  }
  vector<ge::NodePtr> res = {whileNode, tSplitNode};

  return res;
}

Status DynamicRNNGradDFusionPass::DynamicAddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr &while_node,
                                                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/ReduceSumD",
                                                    "ReduceSumD")),
      return FAILED);
  ge::GeTensorDesc reduceInputTensorDescDgate =
           ge::GeTensorDesc(while_node->GetOpDesc()->GetOutputDesc(F_INDEX).GetOriginShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  reduceInputTensorDescDgate.SetOriginShape(while_node->GetOpDesc()->GetOutputDesc(F_INDEX).GetOriginShape());
  reduceInputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  reduceSumDesc->AddInputDesc("input_dgate", reduceInputTensorDescDgate);

  vector<int64_t> reduce_output_origin_dims;
  reduce_output_origin_dims.push_back(W_INDEX);
  reduce_output_origin_dims.push_back(W_INDEX);
  reduce_output_origin_dims.push_back(INIT_H_INDEX*hidden_size);
  ge::GeShape reduce_output_origin_shape(reduce_output_origin_dims);

  ge::GeTensorDesc outputTensorDescDgate = ge::GeTensorDesc(reduce_output_origin_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  outputTensorDescDgate.SetOriginShape(reduce_output_origin_shape);
  outputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDescDgate);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {X_INDEX, W_INDEX});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(reduceSumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                      reduceSumNode->GetName().c_str()),
                    return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(while_node->GetOutDataAnchor(F_INDEX), reduceSumNode->GetInDataAnchor(X_INDEX));
  if (dynamicRNNGradNode->GetOutDataAnchor(W_INDEX)->GetPeerInDataAnchors().size() > X_INDEX) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(W_INDEX)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(X_INDEX), inAnchorPtr);
    }
  }

  return SUCCESS;
}


Status DynamicRNNGradDFusionPass::DynamicAddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (reduceSumDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ReduceSumD",
                                                    "ReduceSumD")),
      return FAILED);
  vector<int64_t> input_dims;
  input_dims.push_back(t_size);
  input_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(W_INDEX));
  input_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(2));
  input_dims.push_back(MASK_INDEX);
  input_dims.push_back(MASK_INDEX);

  vector<int64_t> input_ori_dims;
  input_ori_dims.push_back(t_size);
  input_ori_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(2) * MASK_INDEX);
  input_ori_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(W_INDEX) * MASK_INDEX);

  ge::GeShape input_shape(input_dims);
  ge::GeTensorDesc inputTensorDescMatmul = ge::GeTensorDesc(input_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescMatmul.SetOriginShape(GeShape(input_ori_dims));
  inputTensorDescMatmul.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddInputDesc("input_matmul", inputTensorDescMatmul);

  vector<int64_t> output_dims;
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(W_INDEX)));
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(2)));
  output_dims.push_back(MASK_INDEX);
  output_dims.push_back(MASK_INDEX);

  vector<int64_t> output_ori_dims;
  output_ori_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(2) * MASK_INDEX));
  output_ori_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(X_INDEX).GetShape().GetDim(W_INDEX) * MASK_INDEX));

  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(GeShape(output_ori_dims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {X_INDEX});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(reduceSumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                      reduceSumNode->GetName().c_str()),
                    return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(X_INDEX), reduceSumNode->GetInDataAnchor(X_INDEX));
  if (dynamicRNNGradNode->GetOutDataAnchor(X_INDEX)->GetPeerInDataAnchors().size() > X_INDEX) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(X_INDEX)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(X_INDEX), inAnchorPtr);
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  // get dynamicRNNGradNode
  ge::NodePtr dynamicRNNGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(dynamicRNNGradNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                           "Get DynamicRnnGrad Node Failed, fusion failed."),
                    return FAILED);

  batch_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["x"]).GetShape().GetDim(W_INDEX);
  input_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["x"]).GetShape().GetDim(2);
  hidden_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["h"]).GetShape().GetDim(2);

  if (hidden_size % MASK_INDEX != X_INDEX || input_size % MASK_INDEX != X_INDEX) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is not MASK_INDEX align, will not changed");
    return NOT_CHANGED;
  }

  DynamicRNNGradName = dynamicRNNGradNode->GetName();
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).
                                                                            GetShape().GetDims().size() == 3) {
    vector<int64_t> init_c_dims = {dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).
                                                                                    GetShape().GetDim(W_INDEX),
                                   dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).
                                                                                    GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"])->\
      SetShape(GeShape(init_c_dims));
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"])->
                                                  SetOriginShape(GeShape(init_c_dims));
    std::string reshapeName = DynamicRNNGradName + "initCReshapeNode";
    reshapeInitC = DynamicAddInputReshapeNode(dynamicRNNGradNode, reshapeName,
                          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).Clone(),
                                              graph, newNodes, failStatus);
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["init_c"])->GetPeerOutAnchor()
                            , reshapeInitC->GetInDataAnchor(X_INDEX));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    vector<int64_t> dh_dims = {dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).
                                                                                              GetShape().GetDim(W_INDEX),
                                   dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"])
                                                                                      .GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"])->SetShape(GeShape(dh_dims));
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"])->\
      SetOriginShape(GeShape(dh_dims));
    std::string reshapeDhName = DynamicRNNGradName + "dHReshapeNode";
    reshapeDh = DynamicAddInputReshapeNode(dynamicRNNGradNode, reshapeDhName,
                                dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).Clone(),
                                              graph, newNodes, failStatus);
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                            reshapeDh->GetInDataAnchor(X_INDEX));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).GetShape().GetDims().size() == 3) {
    vector<int64_t> dc_dims = {dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).
                               GetShape().GetDim(W_INDEX),
                                   dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).
                               GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"])->SetShape(GeShape(dc_dims));
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"])->\
      SetOriginShape(GeShape(dc_dims));
    std::string reshapeDcName = DynamicRNNGradName + "dCReshapeNode";
    reshapeDc = DynamicAddInputReshapeNode(dynamicRNNGradNode, reshapeDcName,
                                 dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).Clone(),
                                              graph, newNodes, failStatus);
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["dc"])->GetPeerOutAnchor(),
                            reshapeDc->GetInDataAnchor(X_INDEX));
  }
  if (batch_size == -W_INDEX) {
    batch_nz_size = -W_INDEX;
    batch_start = W_INDEX;
    batch_end = 32;
  } else {
    batch_nz_size = batch_size / MASK_INDEX;
    batch_start = batch_nz_size;
    batch_end = batch_nz_size;
  }
  input_nz_size = input_size / MASK_INDEX;
  hidden_nz_size = hidden_size / MASK_INDEX;
  vector<ge::NodePtr> res_while_node = DynamicAddLSTMInputGradNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  ge::NodePtr while_node = res_while_node[X_INDEX];
  ge::NodePtr tSplitNode = res_while_node[W_INDEX];
  ge::NodePtr subNode = BuildSubNode(dynamicRNNGradNode, tSplitNode, graph, failStatus);
  newNodes.push_back(subNode);
  ge::NodePtr sizeConcatNode = BuildSizeConcatNode(dynamicRNNGradNode, subNode, graph, failStatus);
  newNodes.push_back(sizeConcatNode);
  ge::NodePtr concatNode = DynamicAddConcatHCNode(dynamicRNNGradNode, sizeConcatNode, graph, newNodes, failStatus);
  ge::NodePtr matmulNode = DynamicAddMatmulNode(dynamicRNNGradNode, concatNode, while_node,
                                                graph, newNodes, failStatus);
  DynamicAddDwReduceSumNode(dynamicRNNGradNode, matmulNode, graph, newNodes);
  DynamicAddDbReduceSumNode(dynamicRNNGradNode, while_node, graph, newNodes);
  for (auto inAnchor : dynamicRNNGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  return SUCCESS;
}

REGISTER_PASS("DynamicRNNGradDFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNGradDFusionPass);
}
