/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_grad_d_align_fusion_pass.h"

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

static const int BLOCKSIZE = 16;
static const char* FUSED_NODE = "DynamicRNNGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNGrad";
static map<std::string, int> RNN_GRAD_NODE_INPUT_INDEX = {
    {"x", 0},   {"w", 1},  {"init_c", 5}, {"init_h", 4}, {"h", 6},  {"c", 7},       {"dy", 8},   {"dh", 9},
    {"dc", 10}, {"i", 11}, {"j", 12},     {"f", 13},     {"o", 14}, {"tanhct", 15}, {"mask", 16}};

vector<FusionPattern*> DynamicRNNGradDAlignFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNGradDAlignFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::ComputeGraphPtr DynamicRNNGradDAlignFusionPass::BuildCondGraph(ge::NodePtr& whileNode, int32_t argNum) {
  string condName = DynamicRNNGradName + "cond";
  CompleteGraphBuilder graph_builder(condName, false);
  graph_builder.SetParentNode(whileNode);
  std::string lessName = DynamicRNNGradName + "Less";
  OpDescBuilder op_desc_builder(lessName, "Less");
  GeTensorDesc out_desc(GeShape(), FORMAT_ND, DT_BOOL);
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(14).Clone())
      .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(15).Clone())
      .AddOutput("y", out_desc);
  graph_builder.AddNode(op_desc_builder.Build());
  for (int32_t i = 0; i < 14; i++) {
    graph_builder.SetUselessInput(i);
  }
  graph_builder.SetInput(14, {lessName}, {0});
  graph_builder.SetInput(15, {lessName}, {1});
  graph_builder.SetUselessInput(16);
  graph_builder.SetUselessInput(17);
  graph_builder.AddOutput(lessName, 0);
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = 0; i < argNum; i++) {
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

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::CreateListConstDesc(const std::string& name,
                                                                  std::vector<int64_t> values) {
  OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")), return nullptr);

  GeTensorDesc data_desc(GeShape({static_cast<int64_t>(values.size())}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED((const_value = std::make_shared<ge::GeTensor>(
                               data_desc, reinterpret_cast<uint8_t*>(values.data()), sizeof(int64_t) * values.size())),
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

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::CreateConstDesc(const std::string& name, int32_t value,
                                                              const std::string& dtype) {
  OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")), return nullptr);
  if (const_op_desc == nullptr) {
    return nullptr;
  }
  GeTensorDesc data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(&value), sizeof(int32_t))),
      return nullptr);
  if (dtype == "int64") {
    data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64);
    FUSION_PASS_MAKE_SHARED(
        (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(&value), sizeof(int64_t))),
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

ge::NodePtr DynamicRNNGradDAlignFusionPass::GetDynamicReshapeDxNode(std::string& reshapeNodeName,
                                                                    ge::NodePtr dynamicRNNGradNode,
                                                                    ge::GeTensorDesc inputDesc,
                                                                    ge::GeTensorDesc outputDesc,
                                                                    ge::ComputeGraph& graph, bool& failStatus) {
  auto reshapeOp =
      ge::OperatorFactory::CreateOperator(dynamicRNNGradNode->GetName() + "/" + reshapeNodeName, "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  ge::GeShape inputShape = ge::GeShape({(inputDesc.GetOriginShape().GetDim(1) + 15) / 16, batch_nz_size, 16, 16});
  inputDesc.SetShape(inputShape);
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeShape outputShape = ge::GeShape({-1, (inputDesc.GetOriginShape().GetDim(1) + 15) / 16, batch_nz_size, 16, 16});
  ge::GeShape outputOriShape = ge::GeShape({-1, batch_size, inputDesc.GetOriginShape().GetDim(1)});
  auto outDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outDesc.SetOriginShape(outputOriShape);
  outDesc.SetOriginFormat(ge::FORMAT_ND);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  return myReshape_node;
}

vector<ge::OpDescPtr> DynamicRNNGradDAlignFusionPass::GetDynamicBodyReshapeNode(
    std::string& reshapeNodeName, std::string& reshapeConstNodeName, ge::NodePtr dynamicRNNGradNode,
    ge::GeTensorDesc inputDesc, ge::GeTensorDesc outputDesc, ge::ComputeGraph& graph, bool& failStatus) {
  vector<ge::OpDescPtr> res = {};
  auto reshapeOp = ge::OperatorFactory::CreateOperator(reshapeNodeName, "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return res);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);

  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  int64_t dim1 = 1;
  ge::OpDescPtr const_opdesc = CreateListConstDesc(reshapeConstNodeName, {dim1, 16, 4 * hidden_size});
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op operator error"), return res);
  return {reshape_desc, const_opdesc};
}

vector<ge::OpDescPtr> DynamicRNNGradDAlignFusionPass::GetDynamicBodyDxReshapeNode(
    std::string& reshapeNodeName, std::string& reshapeConstNodeName, ge::NodePtr dynamicRNNGradNode,
    ge::GeTensorDesc inputDesc, ge::GeTensorDesc outputDesc, ge::ComputeGraph& graph, bool& failStatus) {
  vector<ge::OpDescPtr> res = {};
  auto reshapeOp = ge::OperatorFactory::CreateOperator(reshapeNodeName, "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return res);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);

  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  vector<int64_t> outDims = {1, input_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> outOriDims = {1, batch_size, input_size};
  auto outDesc = ge::GeTensorDesc(GeShape(outDims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outDesc.SetOriginShape(GeShape(outOriDims));
  outDesc.SetOriginFormat(FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return res);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  int64_t dim1 = 1;
  ge::OpDescPtr const_opdesc = CreateListConstDesc(reshapeConstNodeName, {dim1, 16, input_size});
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op operator error"), return res);
  return {reshape_desc, const_opdesc};
}

vector<ge::NodePtr> DynamicRNNGradDAlignFusionPass::GetDynamicReshapeNode(
    std::string& reshapeNodeName, ge::NodePtr dynamicRNNGradNode, ge::NodePtr dgateInput, ge::GeTensorDesc outputDesc,
    ge::NodePtr shapeNode, ge::ComputeGraph& graph, bool& failStatus) {
  vector<ge::NodePtr> result = {};
  auto reshapeOp =
      ge::OperatorFactory::CreateOperator(dynamicRNNGradNode->GetName() + "/" + reshapeNodeName, "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return result);
  ge::GeTensorDesc inputDesc = dgateInput->GetOpDesc()->GetOutputDesc(0).Clone();
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();
  ge::GeShape inputShape = ge::GeShape({(inputDesc.GetOriginShape().GetDim(1) + 15) / 16, batch_nz_size, 16, 16});
  inputDesc.SetShape(inputShape);
  inputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return result);
  ge::GeShape shapeShape = ge::GeShape({3});

  auto shapeDesc = ge::GeTensorDesc(shapeShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDesc.SetOriginShape(shapeShape);
  shapeDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeShape outputShape = ge::GeShape({-1, (inputDesc.GetOriginShape().GetDim(1) + 15) / 16, batch_nz_size, 16, 16});
  ge::GeShape outputOriShape = ge::GeShape({-1, batch_size, inputDesc.GetOriginShape().GetDim(1)});
  auto outDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outDesc.SetOriginShape(outputOriShape);
  outDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("shape", shapeDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return result);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", outDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return result);

  std::vector<string> dep_inputs = {"shape"};
  reshape_desc->SetOpInferDepends(dep_inputs);

  ge::GeShape shapeConstShape = ge::GeShape({
      1,
  });
  auto shapeDescConst = ge::GeTensorDesc(shapeConstShape, ge::FORMAT_ND, ge::DT_INT64);
  shapeDescConst.SetOriginShape(shapeConstShape);
  shapeDescConst.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorPtr shapeDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((shapeDescTensor = std::make_shared<ge::GeTensor>(shapeDescConst)), return result);
  vector<int64_t> shapeValue;
  int64_t dim1 = -1;
  shapeValue.push_back(static_cast<int64_t>(dim1));

  shapeDescTensor->SetData(reinterpret_cast<uint8_t*>(shapeValue.data()), shapeValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(shapeDescTensor);
  FUSION_PASS_CHECK(const_opdesc == nullptr, OP_LOGE("Create Const Op desc error"), return result);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);
  FUSION_PASS_CHECK(const_node == nullptr, OP_LOGE("Create const Op operator error"), return result);
  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(myReshape_node == nullptr, OP_LOGE("Create Reshape Op operator error"), return result);
  // add const for body dgate
  ge::GeTensorPtr shapeBodyDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((shapeBodyDescTensor = std::make_shared<ge::GeTensor>(shapeDescConst)), return result);
  vector<int64_t> shapeBodyValue;
  shapeBodyValue.push_back(static_cast<int64_t>(1));

  shapeBodyDescTensor->SetData(reinterpret_cast<uint8_t*>(shapeBodyValue.data()),
                               shapeBodyValue.size() * sizeof(int64_t));
  ge::OpDescPtr const_body_opdesc = ge::OpDescUtils::CreateConstOp(shapeBodyDescTensor);
  ge::NodePtr const_body_node = graph.AddNode(const_body_opdesc);
  FUSION_PASS_CHECK(const_body_node == nullptr, OP_LOGE("Create const Op operator error"), return result);

  ge::GeShape lastConstShape = ge::GeShape({
      1,
  });
  auto lastDescConst = ge::GeTensorDesc(lastConstShape, ge::FORMAT_ND, ge::DT_INT64);
  lastDescConst.SetOriginShape(lastConstShape);
  lastDescConst.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorPtr lastDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((lastDescTensor = std::make_shared<ge::GeTensor>(lastDescConst)), return result);
  vector<int64_t> lastValue;
  int64_t dimLast = 4 * hidden_nz_size * 16;
  lastValue.push_back(static_cast<int64_t>(dimLast));

  lastDescTensor->SetData(reinterpret_cast<uint8_t*>(lastValue.data()), lastValue.size() * sizeof(int64_t));
  ge::OpDescPtr last_const_opdesc = ge::OpDescUtils::CreateConstOp(lastDescTensor);
  ge::NodePtr last_const_node = graph.AddNode(last_const_opdesc);
  FUSION_PASS_CHECK(last_const_node == nullptr, OP_LOGE("Create Const Op operator error"), return result);
  ge::NodePtr tSplitNode =
      BuildTDgateSplit(shapeNode->GetOpDesc()->GetOutputDesc(0).Clone(), dynamicRNNGradNode, graph, failStatus);
  std::string reshapeConcatName = "TDgateConcat";
  ge::NodePtr dxReshapeConcatNode =
      BuildDgateReshapeSizeConcatNode(dynamicRNNGradNode, reshapeConcatName, const_node, graph, failStatus);
  std::string reshapeBodyConcatName = "TDgateBodyConcat";
  ge::NodePtr dgateBodyReshapeConcatNode =
      BuildDgateReshapeSizeConcatNode(dynamicRNNGradNode, reshapeBodyConcatName, const_body_node, graph, failStatus);
  ge::GraphUtils::AddEdge(shapeNode->GetOutDataAnchor(0), tSplitNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), dxReshapeConcatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(1), dxReshapeConcatNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(last_const_node->GetOutDataAnchor(0), dxReshapeConcatNode->GetInDataAnchor(2));

  ge::GraphUtils::AddEdge(dxReshapeConcatNode->GetOutDataAnchor(0), myReshape_node->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(const_body_node->GetOutDataAnchor(0), dgateBodyReshapeConcatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(1), dgateBodyReshapeConcatNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(last_const_node->GetOutDataAnchor(0), dgateBodyReshapeConcatNode->GetInDataAnchor(2));
  return {myReshape_node, dgateBodyReshapeConcatNode};
}

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::GetDynamicLSTMGradCellNode(std::string cellNodeName,
                                                                         ge::NodePtr dynamicRNNGradNode,
                                                                         ge::GeTensorDesc curTDesc,
                                                                         ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::GeTensorDesc inputI = dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["i"]);
  ge::OpDescPtr basicLstmCellStateGradDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (basicLstmCellStateGradDesc = std::make_shared<ge::OpDesc>(cellNodeName, "DynamicLSTMGradCell")),
      failStatus = true;
      return nullptr);
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

  basicLstmCellStateGradDesc->AddInputDesc("mask", dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]));

  vector<int64_t> output_dims;
  output_dims.push_back(batch_size);
  output_dims.push_back(4 * hidden_nz_size * 16);

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

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::GetDynamicMatMulNode(std::string matmulNodeName,
                                                                   ge::NodePtr dynamicRNNGradNode,
                                                                   ge::ComputeGraph& graph, bool& failStatus,
                                                                   ge::GeShape dgateShape) {
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::OpDescPtr lstmBatchMatMulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((lstmBatchMatMulDesc = std::make_shared<ge::OpDesc>(matmulNodeName, "MatMulV2")),
                          failStatus = true;
                          return nullptr);
  // add matmul input
  vector<int64_t> left_tensor_dims = {4 * hidden_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> left_ori_tensor_dims = {batch_size, 4 * hidden_nz_size * 16};
  ge::GeTensorDesc left_tensor_desc = ge::GeTensorDesc(GeShape(left_ori_tensor_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  left_tensor_desc.SetOriginShape(GeShape(left_ori_tensor_dims));
  left_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(4 * hidden_nz_size * 16, 4 * hidden_nz_size * 16));
  left_tensor_desc.SetShapeRange(x1_range);
  lstmBatchMatMulDesc->AddInputDesc("x1", left_tensor_desc);

  ge::GeTensorDesc w_tensor_desc = dynamicRNNGradDesc->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["w"]).Clone();
  vector<int64_t> w_tensor_dims = {4 * hidden_nz_size, input_nz_size + hidden_nz_size, 16, 16};
  vector<int64_t> w_ori_tensor_dims = {(input_nz_size + hidden_nz_size) * 16, 4 * hidden_nz_size * 16};
  w_tensor_desc.SetShape(GeShape(w_ori_tensor_dims));
  w_tensor_desc.SetFormat(FORMAT_ND);
  w_tensor_desc.SetOriginShape(GeShape(w_ori_tensor_dims));

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::make_pair((input_nz_size + hidden_nz_size) * 16, (input_nz_size + hidden_nz_size) * 16));
  x2_range.push_back(std::make_pair(4 * hidden_nz_size * 16, 4 * hidden_nz_size * 16));
  w_tensor_desc.SetShapeRange(x2_range);
  lstmBatchMatMulDesc->AddInputDesc("x2", w_tensor_desc);

  // add matmul output
  vector<int64_t> outputy_dims = {input_nz_size + hidden_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> outputy_ori_dims = {batch_size, (input_nz_size + hidden_nz_size) * 16};
  ge::GeTensorDesc outputy_tensor_desc = ge::GeTensorDesc(GeShape(outputy_ori_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  outputy_tensor_desc.SetOriginShape(GeShape(outputy_ori_dims));
  outputy_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> y1_range;
  y1_range.push_back(std::make_pair(1, -1));
  y1_range.push_back(std::make_pair((input_nz_size + hidden_nz_size) * 16, (input_nz_size + hidden_nz_size) * 16));
  outputy_tensor_desc.SetShapeRange(y1_range);
  lstmBatchMatMulDesc->AddOutputDesc("y", outputy_tensor_desc);
  // attr
  ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x1", false);
  ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x2", true);

  return lstmBatchMatMulDesc;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildSizeConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr& subNode,
                                                                ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TConcat", "ConcatD")),
      failStatus = true;
      return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(0).Clone();
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

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::GeTensorPtr x2DescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((x2DescTensor = std::make_shared<ge::GeTensor>(concatX2Desc)), failStatus = true;
                          return nullptr);
  vector<int32_t> x2Value;
  x2Value.push_back(static_cast<int32_t>(batch_size));
  x2Value.push_back(static_cast<int32_t>(hidden_size));

  x2DescTensor->SetData(reinterpret_cast<uint8_t*>(x2Value.data()), x2Value.size() * sizeof(int32_t));
  ge::OpDescPtr x2OpDesc = ge::OpDescUtils::CreateConstOp(x2DescTensor);

  ge::NodePtr x2Node = graph.AddNode(x2OpDesc);
  FUSION_PASS_CHECK(x2Node == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  ge::GraphUtils::AddEdge(subNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(x2Node->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildDxReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                                         std::string& nodeName, ge::NodePtr& negOneNode,
                                                                         ge::NodePtr& inputSizeNode,
                                                                         ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/" + nodeName, "ConcatD")),
      failStatus = true;
      return nullptr);

  ge::GeTensorDesc x1Desc = negOneNode->GetOpDesc()->GetOutputDesc(0).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);
  concatDesc->AddInputDesc("x1", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({1}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x2", concatX2Desc);

  ge::GeTensorDesc x3Desc = inputSizeNode->GetOpDesc()->GetOutputDesc(0).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);
  concatDesc->AddInputDesc("x3", x3Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 3);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildDgateReshapeSizeConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                                            std::string& nodeName, ge::NodePtr& subNode,
                                                                            ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/" + nodeName, "ConcatD")),
      failStatus = true;
      return nullptr);
  ge::GeTensorDesc x1Desc = subNode->GetOpDesc()->GetOutputDesc(0).Clone();
  x1Desc.SetDataType(ge::DT_INT64);
  x1Desc.SetOriginDataType(ge::DT_INT64);

  concatDesc->AddInputDesc("x1", x1Desc);

  auto concatX2Desc = ge::GeTensorDesc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  concatX2Desc.SetOriginShape(GeShape({1}));
  concatX2Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x2", concatX2Desc);

  auto concatX3Desc = ge::GeTensorDesc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  concatX3Desc.SetOriginShape(GeShape({1}));
  concatX3Desc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddInputDesc("x3", concatX3Desc);

  auto concatYDesc = ge::GeTensorDesc(GeShape({3}), ge::FORMAT_ND, ge::DT_INT64);
  concatYDesc.SetOriginShape(GeShape({3}));
  concatYDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("y", concatYDesc);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 3);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildSubNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr& tSplitNode,
                                                         ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr subDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (subDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TSub", "Sub")),
      failStatus = true;
      return nullptr);
  subDesc->AddInputDesc("x1", tSplitNode->GetOpDesc()->GetOutputDesc(0).Clone());

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
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(0), subNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(subOneNode->GetOutDataAnchor(0), subNode->GetInDataAnchor(1));

  return subNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildTShape(ge::GeTensorDesc xDesc, ge::NodePtr dynamicRNNGradNode,
                                                        ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr shapeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (shapeDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TShape", "Shape")),
      failStatus = true;
      return nullptr);
  shapeDesc->AddInputDesc("x", xDesc);
  vector<int64_t> inputDims = xDesc.GetShape().GetDims();
  ge::GeTensorDesc outputDesc = ge::GeTensorDesc(GeShape({inputDims.size()}), ge::FORMAT_ND, ge::DT_INT32);
  outputDesc.SetOriginShape(GeShape({inputDims.size()}));
  outputDesc.SetOriginFormat(ge::FORMAT_ND);
  shapeDesc->AddOutputDesc("y", outputDesc);
  ge::NodePtr shapeNode = graph.AddNode(shapeDesc);

  return shapeNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildTDgateSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode,
                                                             ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr lstmSplitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TDgateSplit", "SplitVD")),
      failStatus = true;
      return nullptr);
  lstmSplitDesc->AddInputDesc("y", shapeDesc);
  vector<int64_t> dx_dims;
  dx_dims.push_back(1);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_ND, ge::DT_INT32);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("t", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back(1);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape dh_ori_shape(dh_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

  vector<int64_t> last_dims;
  last_dims.push_back(1);
  ge::GeShape last_shape(last_dims);
  ge::GeTensorDesc last_tensor_desc = ge::GeTensorDesc(last_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape last_ori_shape(last_dims);

  last_tensor_desc.SetOriginShape(last_ori_shape);
  last_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("last_dim", last_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(1);
  size_splits.push_back(1);
  size_splits.push_back(1);
  ge::AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(lstmSplitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 3);

  ge::NodePtr splitNode = graph.AddNode(lstmSplitDesc);

  return splitNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::BuildTSplit(ge::GeTensorDesc shapeDesc, ge::NodePtr dynamicRNNGradNode,
                                                        ge::ComputeGraph& graph, bool& failStatus) {
  ge::OpDescPtr lstmSplitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (lstmSplitDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/RnnGrad/TSplit", "SplitVD")),
      failStatus = true;
      return nullptr);
  lstmSplitDesc->AddInputDesc("y", shapeDesc);
  vector<int64_t> dx_dims;
  dx_dims.push_back(1);
  ge::GeShape dx_shape(dx_dims);
  ge::GeShape dx_original_shape(dx_dims);

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_ND, ge::DT_INT32);
  tensor_dx.SetOriginShape(dx_original_shape);
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("t", tensor_dx);

  vector<int64_t> dh_dims;
  dh_dims.push_back(shapeDesc.GetShape().GetDim(0) - 1);
  ge::GeShape dh_shape(dh_dims);
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeShape dh_ori_shape(dh_dims);

  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(1);
  size_splits.push_back(shapeDesc.GetShape().GetDim(0) - 1);
  ge::AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(lstmSplitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);

  ge::NodePtr splitNode = graph.AddNode(lstmSplitDesc);

  return splitNode;
}
vector<ge::OpDescPtr> DynamicRNNGradDAlignFusionPass::GetDynamicSplitNode(
    std::string splitNodeName, std::string splitDimNodeName, std::string splitSizeNodeName,
    ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, bool& failStatus, ge::GeTensorDesc matmulOutputDesc) {
  ge::OpDescPtr lstmSplitDesc = nullptr;
  vector<ge::OpDescPtr> nullResult = {};
  FUSION_PASS_MAKE_SHARED((lstmSplitDesc = std::make_shared<ge::OpDesc>(splitNodeName, "SplitV")), failStatus = true;
                          return nullResult);
  ge::GeShape inputShape = GeShape({(input_nz_size + hidden_nz_size), batch_nz_size, 16, 16});
  matmulOutputDesc.SetShape(inputShape);
  matmulOutputDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  ge::GeShape inputOriShape = GeShape({batch_size, (input_nz_size + hidden_nz_size) * 16});
  matmulOutputDesc.SetOriginShape(inputOriShape);
  matmulOutputDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("x", matmulOutputDesc);

  vector<int64_t> dx_dims{input_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> dx_ori_dims{batch_size, input_size};

  ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(GeShape(dx_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  tensor_dx.SetOriginShape(GeShape(dx_ori_dims));
  tensor_dx.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y0", tensor_dx);

  vector<int64_t> dh_dims{hidden_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> dh_ori_dims{batch_size, hidden_size};
  ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(GeShape(dh_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  dh_tensor_desc.SetOriginShape(GeShape(dh_ori_dims));
  dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("y1", dh_tensor_desc);

  ge::GeShape sizeSplitShape = ge::GeShape({2});
  auto sizeSplitDesc = ge::GeTensorDesc(sizeSplitShape, ge::FORMAT_ND, ge::DT_INT32);
  sizeSplitDesc.SetOriginShape(sizeSplitShape);
  sizeSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("size_splits", sizeSplitDesc);

  ge::GeTensorPtr sizeSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((sizeSplitDescTensor = std::make_shared<ge::GeTensor>(sizeSplitDesc)), failStatus = true;
                          return nullResult);
  vector<int32_t> sizeSplitValue;
  sizeSplitValue.push_back(static_cast<int32_t>(input_size));
  sizeSplitValue.push_back(static_cast<int32_t>(hidden_size));

  sizeSplitDescTensor->SetData(reinterpret_cast<uint8_t*>(sizeSplitValue.data()),
                               sizeSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr sizeSplitOpDesc = ge::OpDescUtils::CreateConstOp(sizeSplitDescTensor);
  sizeSplitOpDesc->SetName(splitSizeNodeName);

  ge::GeShape sizeDimShape = ge::GeShape({1});
  auto dimSplitDesc = ge::GeTensorDesc(sizeDimShape, ge::FORMAT_ND, ge::DT_INT32);
  dimSplitDesc.SetOriginShape(sizeDimShape);
  dimSplitDesc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDesc->AddInputDesc("split_dim", dimSplitDesc);

  ge::GeTensorPtr dimSplitDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((dimSplitDescTensor = std::make_shared<ge::GeTensor>(dimSplitDesc)), failStatus = true;
                          return nullResult);
  vector<int32_t> dimSplitValue;
  dimSplitValue.push_back(static_cast<int32_t>(1));

  dimSplitDescTensor->SetData(reinterpret_cast<uint8_t*>(dimSplitValue.data()), dimSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr dimSplitOpDesc = ge::OpDescUtils::CreateConstOp(dimSplitDescTensor);
  dimSplitOpDesc->SetName(splitDimNodeName);

  ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);
  vector<string> depend_names = {"size_splits", "split_dim"};
  lstmSplitDesc->SetOpInferDepends(depend_names);
  vector<ge::OpDescPtr> result = {lstmSplitDesc, sizeSplitOpDesc, dimSplitOpDesc};

  return result;
}

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::GetDynamicBodyDxConcatNode(std::string cellNodeName,
                                                                         ge::NodePtr dynamicRNNGradNode,
                                                                         ge::ComputeGraph& graph, bool& failStatus,
                                                                         ge::GeTensorDesc splitInputDesc,
                                                                         ge::GeTensorDesc concatOriDesc) {
  ge::OpDescPtr dxConcatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((dxConcatDesc = std::make_shared<ge::OpDesc>(cellNodeName, "ConcatD")), failStatus = true;
                          return nullptr);
  dxConcatDesc->AddInputDesc("x0", splitInputDesc);
  dxConcatDesc->AddInputDesc("x1", concatOriDesc);

  dxConcatDesc->AddOutputDesc("y", concatOriDesc);

  ge::AttrUtils::SetInt(dxConcatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(dxConcatDesc, "N", 2);

  return dxConcatDesc;
}
ge::OpDescPtr DynamicRNNGradDAlignFusionPass::GetDynamicDxConcatNode(std::string cellNodeName,
                                                                     ge::NodePtr dynamicRNNGradNode,
                                                                     ge::ComputeGraph& graph, bool& failStatus,
                                                                     ge::GeTensorDesc splitInputDesc,
                                                                     ge::GeTensorDesc concatOriDesc) {
  ge::OpDescPtr dxConcatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((dxConcatDesc = std::make_shared<ge::OpDesc>(cellNodeName, "ConcatD")), failStatus = true;
                          return nullptr);
  dxConcatDesc->AddInputDesc("x0", splitInputDesc);
  dxConcatDesc->AddInputDesc("x1", concatOriDesc);

  dxConcatDesc->AddOutputDesc("y", concatOriDesc);

  ge::AttrUtils::SetInt(dxConcatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(dxConcatDesc, "N", 2);

  return dxConcatDesc;
}

ge::ComputeGraphPtr DynamicRNNGradDAlignFusionPass::BuildBodyGraph(ge::ComputeGraph& graph, ge::NodePtr& whileNode,
                                                                   int32_t argNum, ge::NodePtr dynamicRNNGradNode,
                                                                   ge::GeTensorDesc concatOriDesc,
                                                                   ge::GeTensorDesc concatDgateOriDesc,
                                                                   bool& failStatus) {
  std::string body_name = DynamicRNNGradName + "Body";
  CompleteGraphBuilder graph_builder(body_name, false);
  graph_builder.SetParentNode(whileNode);
  std::string cellNodeName = DynamicRNNGradName + "DynamicLSTMGradCell";
  ge::OpDescPtr cellNode = GetDynamicLSTMGradCellNode(
      cellNodeName, dynamicRNNGradNode, whileNode->GetOpDesc()->GetInputDesc(14).Clone(), graph, failStatus);
  graph_builder.AddNode(cellNode);
  std::string matmulNodeName = DynamicRNNGradName + "MatMulCell";
  ge::OpDescPtr matmulNode = GetDynamicMatMulNode(matmulNodeName, dynamicRNNGradNode, graph, failStatus,
                                                  cellNode->GetOutputDesc(0).GetShape());
  graph_builder.AddNode(matmulNode);

  // modify w00542958
  std::string transposeNodeName = DynamicRNNGradName + "TransposeCell";
  ge::OpDescPtr TransposeRNNNodeDesc = AddTransposeToRNNNode(transposeNodeName, dynamicRNNGradNode, failStatus);
  graph_builder.AddNode(TransposeRNNNodeDesc);

  std::string splitNodeName = DynamicRNNGradName + "SplitCell";
  std::string splitDimNodeName = DynamicRNNGradName + "SplitDimConst";
  std::string splitSizeNodeName = DynamicRNNGradName + "SplitSizeConst";
  vector<ge::OpDescPtr> result =
      GetDynamicSplitNode(splitNodeName, splitDimNodeName, splitSizeNodeName, dynamicRNNGradNode, graph, failStatus,
                          matmulNode->GetOutputDesc(0).Clone());
  graph_builder.AddNode(result[0]);
  graph_builder.AddNode(result[1]);
  graph_builder.AddNode(result[2]);
  std::string concatDxNodeName = DynamicRNNGradName + "dxConcatCell";
  ge::GeTensorDesc splitInputDesc = result[0]->GetOutputDesc(0).Clone();
  ge::OpDescPtr concatDxDesc =
      GetDynamicDxConcatNode(concatDxNodeName, dynamicRNNGradNode, graph, failStatus, concatOriDesc, concatOriDesc);
  graph_builder.AddNode(concatDxDesc);
  string bodyDxReshapeNodeName = DynamicRNNGradName + "bodyDxReshapeNode";
  string constDxReshapeInputName = DynamicRNNGradName + "dxReshapeConst";
  vector<ge::OpDescPtr> bodyDxReshapeNodes = GetDynamicBodyDxReshapeNode(
      bodyDxReshapeNodeName, constDxReshapeInputName, dynamicRNNGradNode, result[0]->GetOutputDesc(0).Clone(),
      concatDxDesc->GetInputDesc(0).Clone(), graph, failStatus);
  graph_builder.AddNode(bodyDxReshapeNodes[0]);
  std::string dgateConcatNodeName = DynamicRNNGradName + "dgateConcatCell";
  ge::GeTensorDesc dgateInputDesc = cellNode->GetOutputDesc(0).Clone();
  vector<int64_t> dgateShape = {1, dgateInputDesc.GetShape().GetDim(0), dgateInputDesc.GetShape().GetDim(1)};
  dgateInputDesc.SetShape(GeShape(dgateShape));
  dgateInputDesc.SetOriginShape(GeShape(dgateShape));
  ge::OpDescPtr concatDgateDesc = GetDynamicBodyDxConcatNode(dgateConcatNodeName, dynamicRNNGradNode, graph, failStatus,
                                                             dgateInputDesc, concatDgateOriDesc);
  graph_builder.AddNode(concatDgateDesc);

  string bodyReshapeNodeName = DynamicRNNGradName + "bodyDgateReshapeNode";
  string constReshapeInputName = DynamicRNNGradName + "reshapeConst";
  vector<ge::OpDescPtr> bodyDgateReshapeNodes = GetDynamicBodyReshapeNode(
      bodyReshapeNodeName, constReshapeInputName, dynamicRNNGradNode, cellNode->GetOutputDesc(0).Clone(),
      concatDgateDesc->GetInputDesc(0).Clone(), graph, failStatus);
  graph_builder.AddNode(bodyDgateReshapeNodes[0]);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  int64_t idxOri = 1;
  int64_t idxDgate = 0;

  if (direction == "UNIDIRECTIONAL") {
    idxOri = 1;
    idxDgate = 0;
  } else {
    idxOri = 0;
    idxDgate = 1;
  }

  string constName = DynamicRNNGradName + "OneConst";
  graph_builder.AddNode(CreateConstDesc(constName, 1, "int32"));
  std::string addName = DynamicRNNGradName + "Add";
  OpDescBuilder op_desc_builder(addName, "Add");
  op_desc_builder.AddInput("x1", whileNode->GetOpDesc()->GetInputDesc(14).Clone())
      .AddInput("x2", whileNode->GetOpDesc()->GetInputDesc(14).Clone())
      .AddOutput("y", whileNode->GetOpDesc()->GetInputDesc(14).Clone());
  graph_builder.AddNode(op_desc_builder.Build());
  graph_builder.SetInput(2, {cellNodeName}, {0})
      .SetInput(3, {cellNodeName}, {1})
      .SetInput(4, {cellNodeName}, {2})
      .SetInput(5, {cellNodeName}, {3})
      .SetInput(6, {cellNodeName}, {4})
      .SetInput(7, {cellNodeName}, {5})
      .SetInput(8, {cellNodeName}, {6})
      .SetInput(9, {cellNodeName}, {7})
      .SetInput(10, {cellNodeName}, {8})
      .SetInput(11, {cellNodeName}, {9})
      .SetInput(12, {cellNodeName}, {11})
      .SetInput(14, {cellNodeName, addName}, {10, 0});
  graph_builder.SetUselessInput(15);
  graph_builder.SetInput(1, {transposeNodeName}, {0});
  graph_builder.SetInput(0, {concatDxNodeName}, {1});
  graph_builder.SetInput(13, {dgateConcatNodeName}, {idxOri});

  graph_builder.SetInput(16, {bodyDxReshapeNodeName}, {1});
  graph_builder.SetInput(17, {bodyReshapeNodeName}, {1});

  graph_builder.AddDataLink(cellNodeName, 0, matmulNodeName, 0)
      .AddDataLink(transposeNodeName, 0, matmulNodeName, 1)
      .AddDataLink(matmulNodeName, 0, splitNodeName, 0)
      .AddDataLink(splitNodeName, 0, bodyDxReshapeNodeName, 0)
      .AddDataLink(bodyDxReshapeNodeName, 0, concatDxNodeName, 0)
      .AddDataLink(cellNodeName, 0, bodyReshapeNodeName, 0)
      .AddDataLink(bodyReshapeNodeName, 0, dgateConcatNodeName, idxDgate)
      .AddDataLink(splitDimNodeName, 0, splitNodeName, 2)
      .AddDataLink(splitSizeNodeName, 0, splitNodeName, 1)
      .AddDataLink(constName, 0, addName, 1);
  graph_builder.AddOutput(concatDxNodeName, 0);

  for (uint32_t i = 1; i < 5; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  graph_builder.AddOutput(splitNodeName, 1);
  graph_builder.AddOutput(cellNodeName, 1);
  for (uint32_t i = 7; i < 11; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  graph_builder.AddOutput("Data_" + std::to_string(11), 0);
  graph_builder.AddOutput("Data_" + std::to_string(12), 0);
  graph_builder.AddOutput(dgateConcatNodeName, 0);
  graph_builder.AddOutput(addName, 0);
  graph_builder.AddOutput("Data_" + std::to_string(15), 0);

  graph_builder.AddOutput("Data_" + std::to_string(16), 0);
  graph_builder.AddOutput("Data_" + std::to_string(17), 0);
  std::map<uint32_t, uint32_t> input_mapping;
  for (int32_t i = 0; i < argNum; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  std::map<uint32_t, uint32_t> output_mapping;
  for (int32_t i = 0; i < argNum; i++) {
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
  whileNode->GetOpDesc()->AddSubgraphName(DynamicRNNGradName + "Body");
  whileNode->GetOpDesc()->SetSubgraphInstanceName(index, body_name);

  return body_graph;
}

vector<ge::NodePtr> DynamicRNNGradDAlignFusionPass::BuildWhileNodes(
    ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus,
    ge::GeTensorDesc concatOriDesc, ge::GeTensorDesc concatDgateOriDesc, ge::GeTensorDesc curTDesc,
    ge::GeTensorDesc tDesc, ge::GeTensorDesc reshapeDxDesc, ge::GeTensorDesc reshapeDgateDesc) {
  OpDescBuilder op_desc_builder(DynamicRNNGradName + "While_Op", "While");
  OpDescPtr op_desc =
      op_desc_builder.AddInput("input0", concatOriDesc)
          .AddInput("input1", dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["w"]).Clone())
          .AddInput("input2",
                    dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).Clone())
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
          .AddOutput("output12",
                     dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["mask"]).Clone())
          .AddOutput("output13", concatDgateOriDesc)
          .AddOutput("output14", curTDesc)
          .AddOutput("output15", tDesc)
          .AddOutput("output16", reshapeDxDesc)
          .AddOutput("output17", reshapeDgateDesc)
          .Build();
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
  ge::ComputeGraphPtr body_graph =
      BuildBodyGraph(graph, whileNode, 18, dynamicRNNGradNode, concatOriDesc, concatDgateOriDesc, failStatus);
  if ((body_graph == nullptr) || (graph.AddSubgraph(body_graph) != ge::GRAPH_SUCCESS)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add while_body_graph failed.");
    return {};
  }
  auto graphNodes = cond_graph->GetAllNodes();
  auto bodyNodes = body_graph->GetAllNodes();
  vector<ge::NodePtr> result = {};
  for (const NodePtr& node : graphNodes) {
    result.push_back(node);
  }
  for (const NodePtr& node : bodyNodes) {
    result.push_back(node);
  }
  result.push_back(whileNode);
  return result;
}

vector<ge::NodePtr> DynamicRNNGradDAlignFusionPass::BuildT0Graph(ge::NodePtr dynamicRNNGradNode,
                                                                 ge::GeTensorDesc curTDesc, ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes, bool& failStatus) {
  std::string cellNodeName = DynamicRNNGradName + "DynamicLSTMGradCell0";
  ge::OpDescPtr cellNode = GetDynamicLSTMGradCellNode(cellNodeName, dynamicRNNGradNode, curTDesc, graph, failStatus);
  ge::NodePtr t0CellNode = graph.AddNode(cellNode);

  std::string t0MatmulNodeName = DynamicRNNGradName + "MatMulCell0";
  ge::OpDescPtr matmulNode = GetDynamicMatMulNode(t0MatmulNodeName, dynamicRNNGradNode, graph, failStatus,
                                                  cellNode->GetOutputDesc(0).GetShape());
  ge::NodePtr t0MatmulNode = graph.AddNode(matmulNode);

  // modify w00542958
  std::string t0transposeNodeName = DynamicRNNGradName + "T0TransposeCell";
  ge::OpDescPtr t0TransposeRNNDesco = AddTransposeToRNNNode(t0transposeNodeName, dynamicRNNGradNode, failStatus);
  ge::NodePtr t0TransposeRNNNode = graph.AddNode(t0TransposeRNNDesco);

  std::string splitNodeName = DynamicRNNGradName + "SplitCell0";
  std::string splitDimNodeName = DynamicRNNGradName + "SplitDimConst0";
  std::string splitSizeNodeName = DynamicRNNGradName + "SplitSizeConst0";

  vector<ge::OpDescPtr> result =
      GetDynamicSplitNode(splitNodeName, splitDimNodeName, splitSizeNodeName, dynamicRNNGradNode, graph, failStatus,
                          matmulNode->GetOutputDesc(0).Clone());
  ge::NodePtr t0SplitNode = graph.AddNode(result[0]);
  ge::NodePtr t0SizeSplitNode = graph.AddNode(result[1]);
  ge::NodePtr t0DimSplitNode = graph.AddNode(result[2]);

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).GetShape().GetDims().size() ==
      3) {
    ge::GraphUtils::AddEdge(reshapeInitC->GetOutDataAnchor(0), t0CellNode->GetInDataAnchor(0));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(5)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(0));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(7)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(8)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(2));
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeDh->GetOutDataAnchor(0), t0CellNode->GetInDataAnchor(3));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(9)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(3));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    ge::GraphUtils::AddEdge(reshapeDc->GetOutDataAnchor(0), t0CellNode->GetInDataAnchor(4));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(10)->GetPeerOutAnchor(),
                            t0CellNode->GetInDataAnchor(4));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(11)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(12)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(6));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(13)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(7));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(14)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(8));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(15)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(9));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(16)->GetPeerOutAnchor(), t0CellNode->GetInDataAnchor(11));

  string constName = DynamicRNNGradName + "curT0Const";
  ge::OpDescPtr curTConst = CreateConstDesc(constName, 0, "int32");
  ge::NodePtr curTConstNode = graph.AddNode(curTConst);
  FUSION_PASS_CHECK(curTConstNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  t0CellNode->AddLinkFrom(10, curTConstNode);

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                          t0TransposeRNNNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(t0TransposeRNNNode->GetOutDataAnchor(0), t0MatmulNode->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(t0CellNode->GetOutDataAnchor(0), t0MatmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(t0MatmulNode->GetOutDataAnchor(0), t0SplitNode->GetInDataAnchor(0));

  t0SplitNode->AddLinkFrom(1, t0SizeSplitNode);
  t0SplitNode->AddLinkFrom(2, t0DimSplitNode);

  return {t0CellNode, t0SplitNode};
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                                                 ge::NodePtr& while_node, ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul", "BatchMatMul")),
                          failStatus = true;
                          return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescXh =
      ge::GeTensorDesc(concatNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescXh.SetOriginShape(concatNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape());
  inputTensorDescXh.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorDesc inputTensorDescXhTotal = inputTensorDescXh.Clone();
  vector<int64_t> x1NzDims = {t_size, input_nz_size + hidden_nz_size, batch_nz_size, 16, 16};
  inputTensorDescXhTotal.SetShape(GeShape(x1NzDims));
  inputTensorDescXhTotal.SetFormat(ge::FORMAT_FRACTAL_NZ);

  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair((input_size + hidden_size + 15) / 16, (input_nz_size + hidden_nz_size)));
  x1_range.push_back(std::make_pair(batch_start, batch_end));
  x1_range.push_back(std::make_pair(16, 16));
  x1_range.push_back(std::make_pair(16, 16));
  inputTensorDescXhTotal.SetShapeRange(x1_range);
  //{t_dim, batch_dim, hidden_nz_dim * 16 * 4};
  vector<int64_t> x2NzDims = {t_size, hidden_nz_size * 4, batch_nz_size, 16, 16};
  vector<int64_t> x2OriDims = {t_size, batch_size, hidden_nz_size * 4 * 16};
  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(GeShape(x2NzDims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(GeShape(x2OriDims));
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::make_pair(1, -1));
  x2_range.push_back(std::make_pair(hidden_nz_size * 4, hidden_nz_size * 4));
  x2_range.push_back(std::make_pair(batch_start, batch_end));
  x2_range.push_back(std::make_pair(16, 16));
  x2_range.push_back(std::make_pair(16, 16));
  inputTensorDescDgate.SetShapeRange(x2_range);

  matmulDesc->AddInputDesc("x1", inputTensorDescXhTotal);
  matmulDesc->AddInputDesc("x2", inputTensorDescDgate);

  vector<int64_t> outputDims{t_size, input_nz_size * 16 + hidden_nz_size * 16, hidden_nz_size * 16 * 4};
  vector<int64_t> outputNzDims{t_size, hidden_nz_size * 4, input_nz_size + hidden_nz_size, 16, 16};
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputNzDims);

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_range.push_back(std::make_pair(1, -1));
  y_range.push_back(std::make_pair((input_size + hidden_size + 15) / 16, (input_nz_size + hidden_nz_size)));
  y_range.push_back(std::make_pair((hidden_nz_size * 4), (hidden_nz_size * 4)));
  y_range.push_back(std::make_pair(16, 16));
  y_range.push_back(std::make_pair(16, 16));
  outputTensorDesc.SetShapeRange(y_range);

  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(
      matmulNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", matmulNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(matmulNode);
  // Edge
  ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(while_node->GetOutDataAnchor(13), matmulNode->GetInDataAnchor(1));  // dgate

  return matmulNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddConcatNode(ge::NodePtr dynamicRNNGradNode,
                                                                 ge::NodePtr hConcatNode, ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);

  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  // modify to nz
  vector<int64_t> input_x_nz_dims{t_size, input_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> input_x_ori_dims{t_size, batch_nz_size, input_nz_size * 16};
  vector<int64_t> input_inith_nz_dims{t_size, hidden_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> input_inith_ori_dims{t_size, batch_nz_size, hidden_nz_size * 16};
  inputTensorDescX.SetShape(GeShape(input_x_nz_dims));
  inputTensorDescX.SetOriginShape(GeShape(input_x_ori_dims));
  inputTensorDescX.SetFormat(ge::FORMAT_FRACTAL_NZ);
  inputTensorDescH.SetShape(GeShape(input_inith_nz_dims));
  inputTensorDescH.SetOriginShape(GeShape(input_inith_ori_dims));
  inputTensorDescH.SetFormat(ge::FORMAT_FRACTAL_NZ);

  concatDesc->AddInputDesc("x0", inputTensorDescX);
  concatDesc->AddInputDesc("x1", inputTensorDescH);

  vector<int64_t> outputDims{t_size, input_nz_size + hidden_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> outputOriDims{t_size, batch_size, (input_nz_size + hidden_nz_size) * 16};
  ge::GeShape outputShape(outputDims);
  ge::GeShape outputOriShape(outputOriDims);

  ge::GeTensorDesc outputTensorDesc =
      ge::GeTensorDesc(outputShape, ge::FORMAT_FRACTAL_NZ, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputOriShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  concatDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  newNodes.push_back(concatNode);

  ge::NodePtr dxPadNode = AddDxPadNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(), dxPadNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dxPadNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  //  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
  //  concatNode->GetInDataAnchor(0)); ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0),
  //  concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                                                  ge::ComputeGraph& graph,
                                                                  vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/HConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(4);
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();

  vector<int64_t> input_h;
  input_h.push_back(1);
  input_h.push_back(batch_size);
  input_h.push_back(hidden_size);
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  int64_t idxInitH = 0;
  int64_t idxSplit = 1;
  if (direction == "UNIDIRECTIONAL") {
    concatDesc->AddInputDesc("x0", inputTensorDescInitH);
    concatDesc->AddInputDesc("x1", inputTensorDescSplitH);
  } else {
    concatDesc->AddInputDesc("x0", inputTensorDescSplitH);
    concatDesc->AddInputDesc("x1", inputTensorDescInitH);
    idxInitH = 1;
    idxSplit = 0;
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
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
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
        reshapeInitH->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(reshapeInitH->GetOutDataAnchor(0), concatNode->GetInDataAnchor(idxInitH));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                            concatNode->GetInDataAnchor(idxInitH));
  }

  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(idxSplit));

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddSplitNode(ge::NodePtr dynamicRNNGradNode,
                                                                ge::NodePtr& sizeConcatNode, ge::ComputeGraph& graph,
                                                                vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr sliceDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (sliceDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "RNNWeightGrad/Dh/Slice", "Slice")),
      failStatus = true;
      return nullptr);
  ge::GeTensorDesc sliceInputDesc =
      dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["h"]).Clone();
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(1, -1));
  x1_range.push_back(std::make_pair(hidden_size, hidden_size));
  sliceInputDesc.SetShapeRange(x1_range);
  sliceDesc->AddInputDesc("x", sliceInputDesc);

  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(dynamicRNNGradNode->GetOpDesc(), "direction", direction);
  vector<int64_t> output1Dim = {0, 0, 0};
  if (direction == "UNIDIRECTIONAL") {
    output1Dim = {0, 0, 0};
  } else {
    output1Dim = {1, 0, 0};
  }
  GeTensorDesc offsetDesc(GeShape({static_cast<int64_t>(output1Dim.size())}), FORMAT_ND, DT_INT64);
  sliceDesc->AddInputDesc("offsets", offsetDesc);
  ge::OpDescPtr offsetConst = CreateListConstDesc("addWhileHSliceOffset", output1Dim);
  ge::NodePtr offsetNode = graph.AddNode(offsetConst);
  FUSION_PASS_CHECK(offsetNode == nullptr, OP_LOGE("Create Const Op operator error"), return nullptr);
  newNodes.push_back(offsetNode);
  vector<int64_t> output2Dim = {-1, batch_size, hidden_size};

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
                          sliceNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(offsetNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(sizeConcatNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(2));

  return sliceNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddConcatHCNode(ge::NodePtr dynamicRNNGradNode,
                                                                   ge::NodePtr& sizeConcatNode, ge::ComputeGraph& graph,
                                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::NodePtr splitNode = DynamicAddSplitNode(dynamicRNNGradNode, sizeConcatNode, graph, newNodes, failStatus);

  ge::NodePtr hConcatNode = DynamicAddHConcatNode(dynamicRNNGradNode, splitNode, graph, newNodes, failStatus);

  ge::NodePtr concatNode = DynamicAddConcatNode(dynamicRNNGradNode, hConcatNode, graph, newNodes, failStatus);

  return concatNode;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddInputReshapeNode(ge::NodePtr dynamicRNNGradNode,
                                                                       string reshapeName, ge::GeTensorDesc inputDesc,
                                                                       ge::ComputeGraph& graph,
                                                                       vector<ge::NodePtr>& newNodes,
                                                                       bool& failStatus) {
  auto reshapeOp = ge::OperatorFactory::CreateOperator(dynamicRNNGradNode->GetName() + "/" + reshapeName, "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  vector<int64_t> outputReshapeDims = {inputDesc.GetShape().GetDim(1), inputDesc.GetShape().GetDim(2)};
  ge::GeShape outputReshapeShape(outputReshapeDims);

  ge::GeTensorDesc reshapeCellOutputDesc = ge::GeTensorDesc(outputReshapeShape, ge::FORMAT_ND, inputDesc.GetDataType());
  reshapeCellOutputDesc.SetOriginShape(outputReshapeShape);
  reshapeCellOutputDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  FUSION_PASS_CHECK(myReshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Create node error"), return nullptr);

  return myReshape_node;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::DynamicAddInithReshapeNode(ge::NodePtr dynamicRNNGradNode,
                                                                       string reshapeName, ge::GeTensorDesc inputDesc,
                                                                       ge::ComputeGraph& graph,
                                                                       vector<ge::NodePtr>& newNodes,
                                                                       bool& failStatus) {
  auto reshapeOp = ge::OperatorFactory::CreateOperator(dynamicRNNGradNode->GetName() + "/" + reshapeName, "Unsqueeze");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE("Create Reshape Op operator error"), return nullptr);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  vector<int64_t> outputReshapeDims = {1, inputDesc.GetShape().GetDim(0), inputDesc.GetShape().GetDim(1)};
  ge::GeShape outputReshapeShape(outputReshapeDims);

  ge::GeTensorDesc reshapeCellOutputDesc = ge::GeTensorDesc(outputReshapeShape, ge::FORMAT_ND, inputDesc.GetDataType());
  reshapeCellOutputDesc.SetOriginShape(outputReshapeShape);
  reshapeCellOutputDesc.SetOriginFormat(ge::FORMAT_ND);

  // shape range
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.insert(x1_range.begin(), std::make_pair(hidden_size, hidden_size));
  x1_range.insert(x1_range.begin(), std::make_pair(1, -1));
  inputDesc.SetShapeRange(x1_range);
  inputDesc.SetOriginShapeRange(x1_range);

  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateInputDesc("x", inputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Reshape node update outputDesc failed!"), return nullptr);

  // set attr
  ge::AttrUtils::SetListInt(reshape_desc, "axes", {0});

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);

  FUSION_PASS_CHECK(myReshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Create node error"), return nullptr);

  return myReshape_node;
}

vector<ge::NodePtr> DynamicRNNGradDAlignFusionPass::DynamicAddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode,
                                                                                ge::ComputeGraph& graph,
                                                                                vector<ge::NodePtr>& newNodes,
                                                                                bool& failStatus) {
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  ge::OpDescPtr currTConst = CreateConstDesc(DynamicRNNGradName + "currT", 1, "int32");
  ge::NodePtr currTNode = graph.AddNode(currTConst);
  FUSION_PASS_CHECK(currTNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  newNodes.push_back(currTNode);
  ge::NodePtr shapeNode =
      BuildTShape(dynamicRNNGradDesc->GetInputDesc(0).Clone(), dynamicRNNGradNode, graph, failStatus);
  ge::NodePtr tSplitNode =
      BuildTSplit(shapeNode->GetOpDesc()->GetOutputDesc(0).Clone(), dynamicRNNGradNode, graph, failStatus);
  vector<ge::NodePtr> t0Nodes =
      BuildT0Graph(dynamicRNNGradNode, currTConst->GetOutputDesc(0).Clone(), graph, newNodes, failStatus);

  ge::NodePtr cellT0Node = t0Nodes[0];
  ge::NodePtr splitT0Node = t0Nodes[1];

  string dxReshapeNodeName = DynamicRNNGradName + "dxReshapeNode";
  ge::NodePtr dxReshapeNode =
      GetDynamicReshapeDxNode(dxReshapeNodeName, dynamicRNNGradNode, splitT0Node->GetOpDesc()->GetOutputDesc(0).Clone(),
                              cellT0Node->GetOpDesc()->GetOutputDesc(0).Clone(), graph, failStatus);

  string negOneConstName = DynamicRNNGradName + "negOneConst";
  ge::OpDescPtr negOneConst = CreateConstDesc(negOneConstName, -1, "int64");
  ge::NodePtr negOneConstNode = graph.AddNode(negOneConst);
  FUSION_PASS_CHECK(negOneConstNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});

  string inputSizeConstName = DynamicRNNGradName + "inputSizeConst";
  ge::OpDescPtr inputSizeConst = CreateConstDesc(inputSizeConstName, input_size, "int64");
  ge::NodePtr inputSizeConstConstNode = graph.AddNode(inputSizeConst);
  FUSION_PASS_CHECK(inputSizeConstConstNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});

  std::string reshapeDxNodeName = "TDxConcat";
  ge::NodePtr dxReshapeConcatNode = BuildDxReshapeSizeConcatNode(dynamicRNNGradNode, reshapeDxNodeName, negOneConstNode,
                                                                 inputSizeConstConstNode, graph, failStatus);
  newNodes.push_back(dxReshapeConcatNode);
  ge::GeTensorDesc concatOriDesc = t0Nodes[1]->GetOpDesc()->GetOutputDesc(0).Clone();
  concatOriDesc.SetFormat(ge::FORMAT_ND);
  vector<int64_t> dxConcatDims = {-1, input_nz_size, batch_nz_size, 16, 16};
  vector<int64_t> dxOriConcatDims = {-1, batch_size, input_size};

  concatOriDesc.SetShape(GeShape(dxOriConcatDims));
  concatOriDesc.SetOriginShape(GeShape(dxOriConcatDims));

  ge::GeTensorDesc concatDgateOriDesc = t0Nodes[0]->GetOpDesc()->GetOutputDesc(0).Clone();
  vector<int64_t> dgateConcatDims = {-1, batch_size, concatDgateOriDesc.GetShape().GetDim(1)};
  concatDgateOriDesc.SetShape(GeShape(dgateConcatDims));
  concatDgateOriDesc.SetOriginShape(GeShape(dgateConcatDims));

  ge::OpDescPtr totalTConst = CreateConstDesc(DynamicRNNGradName + "totalT", 2, "int32");
  ge::NodePtr totalTNode = graph.AddNode(totalTConst);
  FUSION_PASS_CHECK(totalTNode == nullptr, OP_LOGE("Create Const Op operator error"), return {});
  newNodes.push_back(totalTNode);

  std::string reshapeBodyDxNodeName = "TDxBodyConcat";
  ge::NodePtr dxBodyReshapeConcatNode = BuildDxReshapeSizeConcatNode(
      dynamicRNNGradNode, reshapeBodyDxNodeName, negOneConstNode, inputSizeConstConstNode, graph, failStatus);
  newNodes.push_back(dxBodyReshapeConcatNode);
  vector<ge::NodePtr> whileNodes =
      BuildWhileNodes(dynamicRNNGradNode, graph, newNodes, failStatus, concatOriDesc, concatDgateOriDesc,
                      currTConst->GetOutputDesc(0).Clone(), totalTConst->GetOutputDesc(0).Clone(),
                      dxBodyReshapeConcatNode->GetOpDesc()->GetOutputDesc(0).Clone(),
                      dxBodyReshapeConcatNode->GetOpDesc()->GetOutputDesc(0).Clone());

  int whileNodeSize = whileNodes.size();
  for (int i = 0; i < whileNodeSize; i++) {
    newNodes.push_back(whileNodes[i]);
  }

  string reshapeNodeName = "dgateReshapeNode";
  vector<ge::NodePtr> dgateReshapeResult =
      GetDynamicReshapeNode(reshapeNodeName, dynamicRNNGradNode, cellT0Node,
                            whileNodes[0]->GetOpDesc()->GetOutputDesc(13).Clone(), shapeNode, graph, failStatus);
  ge::NodePtr dgateReshapeNode = dgateReshapeResult[0];
  ge::NodePtr dgateBodyReshapeConcatNode = dgateReshapeResult[1];
  ge::NodePtr whileNode = whileNodes[whileNodeSize - 1];
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(), shapeNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(shapeNode->GetOutDataAnchor(0), tSplitNode->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(negOneConstNode->GetOutDataAnchor(0), dxReshapeConcatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(inputSizeConstConstNode->GetOutDataAnchor(0), dxReshapeConcatNode->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(1), dxReshapeConcatNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dxReshapeConcatNode->GetOutDataAnchor(0), dxReshapeNode->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(splitT0Node->GetOutDataAnchor(0), dxReshapeNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dxReshapeNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(1));

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).GetShape().GetDims().size() ==
      3) {
    ge::GraphUtils::AddEdge(reshapeInitC->GetOutDataAnchor(0), whileNode->GetInDataAnchor(2));
  } else {
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(5)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(2));
  }
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(7)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(8)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(4));

  ge::GraphUtils::AddEdge(splitT0Node->GetOutDataAnchor(1), whileNode->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(cellT0Node->GetOutDataAnchor(1), whileNode->GetInDataAnchor(6));

  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(11)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(7));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(12)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(8));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(13)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(9));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(14)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(10));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(15)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(11));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(16)->GetPeerOutAnchor(), whileNode->GetInDataAnchor(12));
  ge::GraphUtils::AddEdge(cellT0Node->GetOutDataAnchor(0), dgateReshapeNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dgateReshapeNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(13));
  ge::GraphUtils::AddEdge(currTNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(14));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(15));

  ge::GraphUtils::AddEdge(negOneConstNode->GetOutDataAnchor(0), dxBodyReshapeConcatNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(inputSizeConstConstNode->GetOutDataAnchor(0), dxBodyReshapeConcatNode->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(tSplitNode->GetOutDataAnchor(1), dxBodyReshapeConcatNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dxBodyReshapeConcatNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(16));
  ge::GraphUtils::AddEdge(dgateBodyReshapeConcatNode->GetOutDataAnchor(0), whileNode->GetInDataAnchor(17));
  if (dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  if (dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(5), inAnchorPtr);
    }
  }

  if (dynamicRNNGradNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(4)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(whileNode->GetOutDataAnchor(6), inAnchorPtr);
    }
  }
  vector<ge::NodePtr> res = {whileNode, tSplitNode};

  return res;
}

Status DynamicRNNGradDAlignFusionPass::DynamicAddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode,
                                                                 ge::NodePtr& while_node, ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((reduceSumDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/ReduceSumD", "ReduceSumD")),
                          return FAILED);
  ge::GeTensorDesc reduceInputTensorDescDgate =
      ge::GeTensorDesc(while_node->GetOpDesc()->GetOutputDesc(13).GetOriginShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  reduceInputTensorDescDgate.SetOriginShape(while_node->GetOpDesc()->GetOutputDesc(13).GetOriginShape());
  reduceInputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  reduceSumDesc->AddInputDesc("input_dgate", reduceInputTensorDescDgate);

  vector<int64_t> reduce_output_dims{hidden_nz_size * 4 * 16};
  vector<int64_t> reduce_output_origin_dims{hidden_size * 4};
  ge::GeTensorDesc outputTensorDescDgate =
      ge::GeTensorDesc(GeShape(reduce_output_dims), ge::FORMAT_ND_RNN_BIAS, ge::DT_FLOAT16);
  outputTensorDescDgate.SetOriginShape(GeShape(reduce_output_origin_dims));
  outputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDescDgate);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0, 1});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);
  ge::AttrUtils::SetInt(reduceSumDesc, "hidden_size", hidden_size);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(
      reduceSumNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", reduceSumNode->GetName().c_str()),
      return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(while_node->GetOutDataAnchor(13), reduceSumNode->GetInDataAnchor(0));
  if (dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradDAlignFusionPass::DynamicAddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                                                 ge::ComputeGraph& graph,
                                                                 vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((reduceSumDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ReduceSumD", "ReduceSumD")),
                          return FAILED);
  vector<int64_t> input_dims;
  input_dims.push_back(t_size);
  input_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1));
  input_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2));
  input_dims.push_back(16);
  input_dims.push_back(16);

  vector<int64_t> input_ori_dims;
  input_ori_dims.push_back(t_size);
  input_ori_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) * 16);
  input_ori_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1) * 16);

  ge::GeShape input_shape(input_dims);
  ge::GeTensorDesc inputTensorDescMatmul = ge::GeTensorDesc(input_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescMatmul.SetOriginShape(GeShape(input_ori_dims));
  inputTensorDescMatmul.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddInputDesc("input_matmul", inputTensorDescMatmul);

  vector<int64_t> output_dims;
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1)));
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2)));
  output_dims.push_back(16);
  output_dims.push_back(16);

  vector<int64_t> output_ori_dims;
  output_ori_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) * 16));
  output_ori_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1) * 16));

  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(GeShape(output_ori_dims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(
      reduceSumNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", reduceSumNode->GetName().c_str()),
      return FAILED);
  newNodes.push_back(reduceSumNode);

  // add transpose node for nz to fracal_zn_rnn
  bool failStatus = false;
  ge::NodePtr transposeNode = AddTransposeNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(
      failStatus,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddTransposeNode:check failed, fusion failed."),
      return FAILED);

  // Edge
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), transposeNode->GetInDataAnchor(0));

  if (dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }
  return SUCCESS;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::AddTransposeNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create transposed desc
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/transposeD", "TransposeD")),
                          failStatus = true;
                          return nullptr);

  // attr
  vector<int32_t> permValue = {1, 0, 3, 2};
  ge::AttrUtils::SetListInt(transposeDesc, "perm", permValue);
  ge::AttrUtils::SetInt(transposeDesc, "input_size", input_size);
  ge::AttrUtils::SetInt(transposeDesc, "hidden_size", hidden_size);

  // input
  vector<int64_t> tran_input_zn_dims{hidden_nz_size * 4, input_nz_size + hidden_nz_size, 16, 16};
  vector<int64_t> tran_input_ori_zn_dims{(input_nz_size + hidden_nz_size) * 16, hidden_nz_size * 4 * 16};
  ge::GeTensorDesc transInputDesc =
      ge::GeTensorDesc(GeShape(tran_input_zn_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  transInputDesc.SetOriginShape(GeShape(tran_input_ori_zn_dims));
  transInputDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddInputDesc("x", transInputDesc);

  // output
  vector<int64_t> tran_output_zn_dims{input_nz_size + hidden_nz_size, hidden_nz_size * 4, 16, 16};
  vector<int64_t> tran_out_ori_zn_dims{input_size + hidden_size, hidden_size * 4};
  ge::GeTensorDesc transOutDesc =
      ge::GeTensorDesc(GeShape(tran_output_zn_dims), ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16);
  transOutDesc.SetOriginShape(GeShape(tran_out_ori_zn_dims));
  transOutDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddOutputDesc("y", transOutDesc);

  // create transpose node
  ge::NodePtr transposeNode = graph.AddNode(transposeDesc);
  FUSION_PASS_CHECK(transposeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   transposeNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(transposeNode);

  return transposeNode;
}

ge::OpDescPtr DynamicRNNGradDAlignFusionPass::AddTransposeToRNNNode(std::string transposeNodeName,
                                                                    ge::NodePtr dynamicRNNGradNode, bool& failStatus) {
  // create transposed desc
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(transposeNodeName, "TransposeD")),
                          failStatus = true;
                          return nullptr);

  // attr
  vector<int32_t> permValue = {1, 0, 3, 2};
  ge::AttrUtils::SetListInt(transposeDesc, "perm", permValue);
  ge::AttrUtils::SetInt(transposeDesc, "input_size", input_size);
  ge::AttrUtils::SetInt(transposeDesc, "hidden_size", hidden_size);

  // input
  vector<int64_t> tran_intput_zn_dims{input_nz_size + hidden_nz_size, hidden_nz_size * 4, 16, 16};
  vector<int64_t> tran_input_ori_zn_dims{input_size + hidden_size, hidden_size * 4};
  ge::GeTensorDesc transInputDesc =
      ge::GeTensorDesc(GeShape(tran_intput_zn_dims), ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16);
  transInputDesc.SetOriginShape(GeShape(tran_input_ori_zn_dims));
  transInputDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddInputDesc("x", transInputDesc);

  // output
  vector<int64_t> tran_output_nz_dims{hidden_nz_size * 4, input_nz_size + hidden_nz_size, 16, 16};
  vector<int64_t> tran_output_ori_nz_dims{(input_nz_size + hidden_nz_size) * 16, hidden_nz_size * 4 * 16};
  ge::GeTensorDesc transOutputDesc =
      ge::GeTensorDesc(GeShape(tran_output_nz_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  transOutputDesc.SetOriginShape(GeShape(tran_output_ori_nz_dims));
  transOutputDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddOutputDesc("y", transOutputDesc);

  return transposeDesc;
}

ge::NodePtr DynamicRNNGradDAlignFusionPass::AddDxPadNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                         vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create pad desc
  ge::OpDescPtr dxPadDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((dxPadDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "Dx/pad", "Pad")),
                          failStatus = true;
                          return nullptr);

  // input
  vector<int64_t> pad_input_dims{t_size, batch_size, input_size};
  ge::GeTensorDesc padInputDesc = ge::GeTensorDesc(GeShape(pad_input_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  padInputDesc.SetOriginShape(GeShape(pad_input_dims));
  padInputDesc.SetOriginFormat(ge::FORMAT_ND);
  dxPadDesc->AddInputDesc("x", padInputDesc);

  // input2 padding
  vector<int64_t> pad_input2_dims{3, 2};
  ge::GeTensorDesc padInput2Desc = ge::GeTensorDesc(GeShape(pad_input2_dims), ge::FORMAT_ND, ge::DT_INT32);
  padInput2Desc.SetOriginShape(GeShape(pad_input2_dims));
  padInput2Desc.SetOriginFormat(ge::FORMAT_ND);
  dxPadDesc->AddInputDesc("paddings", padInput2Desc);

  // output
  vector<int64_t> pad_out_dims{t_size, batch_size, input_nz_size * 16};
  ge::GeTensorDesc padOutDesc = ge::GeTensorDesc(GeShape(pad_out_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  padOutDesc.SetOriginShape(GeShape(pad_out_dims));
  padOutDesc.SetOriginFormat(ge::FORMAT_ND);
  dxPadDesc->AddOutputDesc("y", padOutDesc);

  vector<string> depend_names = {"paddings"};
  dxPadDesc->SetOpInferDepends(depend_names);

  // create pad node
  ge::NodePtr dxPadNode = graph.AddNode(dxPadDesc);
  FUSION_PASS_CHECK(dxPadNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   dxPadNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(dxPadNode);

  // create const node for input2_paddings
  ge::GeTensorPtr padingsConstDescTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((padingsConstDescTensor = std::make_shared<ge::GeTensor>(padInput2Desc)), failStatus = true;
                          return nullptr);

  int32_t padValueforInputSize = input_nz_size * 16 - input_size;
  vector<int32_t> sizeSplitValue{0, 0, 0, 0, 0, padValueforInputSize};
  padingsConstDescTensor->SetData(reinterpret_cast<uint8_t*>(sizeSplitValue.data()),
                                  sizeSplitValue.size() * sizeof(int32_t));
  ge::OpDescPtr padingsConstOpDesc = ge::OpDescUtils::CreateConstOp(padingsConstDescTensor);

  ge::NodePtr padingsConstNode = graph.AddNode(padingsConstOpDesc);
  FUSION_PASS_CHECK(padingsConstNode == nullptr, OP_LOGE("Create Const Op for pad operator error"), return nullptr);
  newNodes.push_back(padingsConstNode);
  ge::GraphUtils::AddEdge(padingsConstNode->GetOutDataAnchor(0), dxPadNode->GetInDataAnchor(1));

  return dxPadNode;
}

Status DynamicRNNGradDAlignFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  // get dynamicRNNGradNode
  ge::NodePtr dynamicRNNGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(dynamicRNNGradNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get DynamicRnnGrad Node Failed, fusion failed."), return FAILED);

  batch_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["x"]).GetShape().GetDim(1);
  input_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["x"]).GetShape().GetDim(2);
  hidden_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["h"]).GetShape().GetDim(2);

  if (hidden_size % 16 == 0 && input_size % 16 == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is 16 align, will not changed.");
    return NOT_CHANGED;
  }

  DynamicRNNGradName = dynamicRNNGradNode->GetName();
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).GetShape().GetDims().size() ==
      3) {
    vector<int64_t> init_c_dims = {
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).GetShape().GetDim(1),
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()
        ->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"])
        ->SetShape(GeShape(init_c_dims));
    dynamicRNNGradNode->GetOpDesc()
        ->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"])
        ->SetOriginShape(GeShape(init_c_dims));
    std::string reshapeName = DynamicRNNGradName + "initCReshapeNode";
    reshapeInitC = DynamicAddInputReshapeNode(
        dynamicRNNGradNode, reshapeName,
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["init_c"]).Clone(), graph, newNodes,
        failStatus);
    ge::GraphUtils::AddEdge(
        dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["init_c"])->GetPeerOutAnchor(),
        reshapeInitC->GetInDataAnchor(0));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDims().size() == 3) {
    vector<int64_t> dh_dims = {
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDim(1),
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"])->SetShape(GeShape(dh_dims));
    dynamicRNNGradNode->GetOpDesc()
        ->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"])
        ->SetOriginShape(GeShape(dh_dims));
    std::string reshapeDhName = DynamicRNNGradName + "dHReshapeNode";
    reshapeDh = DynamicAddInputReshapeNode(
        dynamicRNNGradNode, reshapeDhName,
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dh"]).Clone(), graph, newNodes,
        failStatus);
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["dh"])->GetPeerOutAnchor(),
                            reshapeDh->GetInDataAnchor(0));
  }
  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).GetShape().GetDims().size() == 3) {
    vector<int64_t> dc_dims = {
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).GetShape().GetDim(1),
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).GetShape().GetDim(2)};
    dynamicRNNGradNode->GetOpDesc()->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"])->SetShape(GeShape(dc_dims));
    dynamicRNNGradNode->GetOpDesc()
        ->MutableInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"])
        ->SetOriginShape(GeShape(dc_dims));
    std::string reshapeDcName = DynamicRNNGradName + "dCReshapeNode";
    reshapeDc = DynamicAddInputReshapeNode(
        dynamicRNNGradNode, reshapeDcName,
        dynamicRNNGradNode->GetOpDesc()->GetInputDesc(RNN_GRAD_NODE_INPUT_INDEX["dc"]).Clone(), graph, newNodes,
        failStatus);
    ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(RNN_GRAD_NODE_INPUT_INDEX["dc"])->GetPeerOutAnchor(),
                            reshapeDc->GetInDataAnchor(0));
  }
  if (batch_size == -1) {
    batch_nz_size = -1;
    batch_start = 1;
    batch_end = 32;
  } else {
    batch_nz_size = (batch_size + 15) / 16;
    batch_start = batch_nz_size;
    batch_end = batch_nz_size;
  }
  input_nz_size = (input_size + 15) / 16;
  hidden_nz_size = (hidden_size + 15) / 16;
  vector<ge::NodePtr> res_while_node = DynamicAddLSTMInputGradNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  ge::NodePtr while_node = res_while_node[0];
  ge::NodePtr tSplitNode = res_while_node[1];
  ge::NodePtr subNode = BuildSubNode(dynamicRNNGradNode, tSplitNode, graph, failStatus);
  newNodes.push_back(subNode);
  ge::NodePtr sizeConcatNode = BuildSizeConcatNode(dynamicRNNGradNode, subNode, graph, failStatus);
  newNodes.push_back(sizeConcatNode);
  ge::NodePtr concatNode = DynamicAddConcatHCNode(dynamicRNNGradNode, sizeConcatNode, graph, newNodes, failStatus);
  ge::NodePtr matmulNode =
      DynamicAddMatmulNode(dynamicRNNGradNode, concatNode, while_node, graph, newNodes, failStatus);
  DynamicAddDwReduceSumNode(dynamicRNNGradNode, matmulNode, graph, newNodes);
  DynamicAddDbReduceSumNode(dynamicRNNGradNode, while_node, graph, newNodes);
  for (auto inAnchor : dynamicRNNGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  return SUCCESS;
}

REGISTER_PASS("DynamicRNNGradDAlignFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNGradDAlignFusionPass);
}  // namespace fe
