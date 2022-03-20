/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNV2Grad fusion pass(DynamicRNNV2Grad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_v2_grad_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "fp16_t.hpp"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;

namespace {
const char* FUSED_NODE = "DynamicRNNV2Grad";
const std::string PATTERN_FUSEDNODE = "DynamicRNNV2Grad";
const int64_t UNKNOWN_SHAPE = -1;

const int IDX_X = 0;
const int IDX_WX = 1;
const int IDX_WH = 2;
const int IDX_Y = 3;
const int IDX_H0 = 4;
const int IDX_C0 = 5;
const int IDX_H = 6;
const int IDX_C = 7;
const int IDX_DY = 8;
const int IDX_DH = 9;
const int IDX_DC = 10;
const int IDX_I = 11;
const int IDX_J = 12;
const int IDX_F = 13;
const int IDX_O = 14;
const int IDX_TANH = 15;
const int IDX_SEQ = 16;
const int IDX_WCI = 17;
const int IDX_WCF = 18;
const int IDX_WCO = 19;
const int IDX_MASK = 20;

const int GRAD_OUT_IDX_DWX = 0;
const int GRAD_OUT_IDX_DWH = 1;
const int GRAD_OUT_IDX_DB = 2;
const int GRAD_OUT_IDX_DX = 3;
const int GRAD_OUT_IDX_DH0 = 4;
const int GRAD_OUT_IDX_DC0 = 5;

const int CELL_C0 = 0;
const int CELL_C = 1;
const int CELL_DY = 2;
const int CELL_DH = 3;
const int CELL_DC = 4;
const int CELL_I = 5;
const int CELL_J = 6;
const int CELL_F = 7;
const int CELL_O = 8;
const int CELL_TANH = 9;
const int CELL_T = 10;
const int CELL_MASK = 11;

map<std::string, int> RNN_GRAD_INPUT_INDEX = {
    {"x", IDX_X},         {"wx", IDX_WX},       {"wh", IDX_WH},   {"h0", IDX_H0},   {"c0", IDX_C0},
    {"dy", IDX_DY},       {"dh", IDX_DH},       {"dc", IDX_DC},   {"y", IDX_Y},     {"h", IDX_H},
    {"c", IDX_C},         {"i", IDX_I},         {"j", IDX_J},     {"f", IDX_F},     {"o", IDX_O},
    {"tanhct", IDX_TANH}, {"seq_len", IDX_SEQ}, {"wci", IDX_WCI}, {"wcf", IDX_WCF}, {"wco", IDX_WCO},
    {"mask", IDX_MASK}};

map<std::string, int> LSTM_GRAD_CELL_INPUT_INDEX = {
    {"c0", CELL_C0}, {"c", CELL_C}, {"dy", CELL_DY}, {"dh", CELL_DH},       {"dc", CELL_DC},     {"i", CELL_I},
    {"j", CELL_J},   {"f", CELL_F}, {"o", CELL_O},   {"tanhct", CELL_TANH}, {"t_state", CELL_T}, {"mask", CELL_MASK}};
}  // namespace

namespace fe {
vector<FusionPattern*> DynamicRNNV2GradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNV2GradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicRNNV2GradFusionPass::SetTensorDescription(ge::GeTensorDesc& desc, const vector<int64_t>& dims,
                                                      const ge::Format& format, const ge::DataType& dtype) const {
  ge::GeShape shape(dims);
  desc.SetShape(shape);
  desc.SetDataType(dtype);
  desc.SetFormat(format);
  desc.SetOriginShape(shape);
  desc.SetOriginDataType(dtype);
  desc.SetOriginFormat(format);

  constexpr int64_t unknown_shape = -1;
  auto found = std::find(dims.begin(), dims.end(), unknown_shape);
  if (found == dims.end()) {
    return;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  constexpr int64_t shape_upper = -1;
  for (size_t i = 0; i < dims.size(); i++) {
    if (dims[i] == unknown_shape) {
      range.push_back(std::pair<int64_t, int64_t>(1, shape_upper));
    } else {
      range.push_back(std::pair<int64_t, int64_t>(dims[i], dims[i]));
    }
  }
  desc.SetShapeRange(range);
  return;
}

ge::NodePtr DynamicRNNV2GradFusionPass::AddNewNode(ge::ComputeGraph& graph, ge::OpDescPtr& op_desc,
                                                   vector<ge::NodePtr>& new_nodes) const {
  ge::NodePtr node = graph.AddNode(op_desc);
  if (!node) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add new node failed.");
    return nullptr;
  }
  new_nodes.push_back(node);
  return node;
}

ge::OpDescPtr DynamicRNNV2GradFusionPass::CreateScaleConstDesc(const std::string& name, int32_t value) {
  ge::OpDescPtr const_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((const_op_desc = std::make_shared<ge::OpDesc>(name, "Const")), return nullptr);
  if (const_op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_op_desc is nullptr.");
    return nullptr;
  }
  ge::GeTensorDesc data_desc = GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32);
  ge::GeTensorPtr const_value = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t*>(&value), sizeof(int32_t))),
      return nullptr);
  if (const_value == nullptr) {
    return nullptr;
  }
  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "SetTensor of const node failed.");
    return nullptr;
  }
  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "set output of const_op_desc failed.");
    return nullptr;
  }
  return const_op_desc;
}

template <class T>
ge::NodePtr DynamicRNNV2GradFusionPass::CreateConstNode(const std::string& name, ge::GeTensorDesc& tensor_desc,
                                                        ge::Operator::OpListInt& const_data, int32_t length,
                                                        ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) {
  ge::OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>(name, "Const");
  if (const_op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_op_desc is nullptr.");
    return nullptr;
  }
  ge::GeTensorPtr const_value =
      std::make_shared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(const_data.data()), length * sizeof(T));
  if (const_value == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_op_desc is nullptr.");
    return nullptr;
  }
  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "SetTensor of const node failed.");
    return nullptr;
  }
  if (const_op_desc->AddOutputDesc("y", tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "set output of const_op_desc failed.");
    return nullptr;
  }

  return AddNewNode(graph, const_op_desc, new_nodes);
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDxhMatMulNode(const ge::NodePtr& lstm_cell_node,
                                                             const ge::NodePtr& w_concat_node, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dxh/MatMul";
  ge::OpDescPtr matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmul_desc = std::make_shared<ge::OpDesc>(node_name, "MatMulV2")), return nullptr);

  // left tensor is dgate
  constexpr int gate_num = 4;
  ge::GeTensorDesc dgate_desc;
  vector<int64_t> dgate_dims = {batch_size, gate_num * hidden_size};
  SetTensorDescription(dgate_desc, dgate_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x1", dgate_desc);

  // right tensor is w.T
  ge::GeTensorDesc w_tensor_desc;
  vector<int64_t> w_tensor_dims = {input_size + hidden_size, gate_num * hidden_size};
  SetTensorDescription(w_tensor_desc, w_tensor_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x2", w_tensor_desc);

  // add matmul output
  ge::GeTensorDesc dxy_tensor_desc;
  vector<int64_t> dxy_dims = {batch_size, (input_size + hidden_size)};
  SetTensorDescription(dxy_tensor_desc, dxy_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddOutputDesc("y", dxy_tensor_desc);

  // attr
  ge::AttrUtils::SetBool(matmul_desc, "transpose_x1", false);
  ge::AttrUtils::SetBool(matmul_desc, "transpose_x2", true);

  ge::NodePtr matmul_node = AddNewNode(graph, matmul_desc, new_nodes);
  ge::GraphUtils::AddEdge(lstm_cell_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(w_concat_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
  return matmul_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDwMatMulNode(const ge::NodePtr& xh_node, const ge::NodePtr& cell_node,
                                                            ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dw/MatMul";
  ge::OpDescPtr matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmul_desc = std::make_shared<ge::OpDesc>(node_name, "MatMulV2")), return nullptr);

  // left tensor is xh.T
  ge::GeTensorDesc xh_desc;
  vector<int64_t> xh_dims = {batch_size, input_size + hidden_size};
  SetTensorDescription(xh_desc, xh_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x1", xh_desc);

  // left tensor is dgate
  constexpr int gate_num = 4;
  ge::GeTensorDesc dgate_desc;
  vector<int64_t> dgate_dims = {batch_size, gate_num * hidden_size};
  SetTensorDescription(dgate_desc, dgate_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x2", dgate_desc);

  // add matmul output
  ge::GeTensorDesc dw_tensor_desc;
  vector<int64_t> dw_dims = {input_size + hidden_size, gate_num * hidden_size};
  SetTensorDescription(dw_tensor_desc, dw_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddOutputDesc("y", dw_tensor_desc);

  ge::AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  ge::AttrUtils::SetBool(matmul_desc, "transpose_x2", false);

  ge::NodePtr matmul_node = AddNewNode(graph, matmul_desc, new_nodes);
  ge::GraphUtils::AddEdge(xh_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cell_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
  return matmul_node;
}

ge::OpDescPtr DynamicRNNV2GradFusionPass::GetDynamicLSTMGradCellDesc(ge::OpDescPtr& fused_desc,
                                                                     ge::GeTensorDesc& curT_desc) {
  std::string cell_node_name = grad_name + "/DynamicLSTMGradCell0";
  ge::OpDescPtr basic_cell_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((basic_cell_desc = std::make_shared<ge::OpDesc>(cell_node_name, "DynamicLSTMGradCell")),
                          return nullptr);
  if (basic_cell_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "basic_cell_desc is nullptr.");
    return nullptr;
  }

  constexpr int64_t time_step = 1;  
  ge::GeTensorDesc input_desc;
  vector<int64_t> input_dims = {time_step, batch_size, hidden_size};
  SetTensorDescription(input_desc, input_dims, ge::FORMAT_ND, state_type);

  for (int i=0;i<CELL_T;i++){
    basic_cell_desc->AddInputDesc(i, input_desc);
  }
  basic_cell_desc->AddInputDesc(CELL_T, curT_desc);

  ge::GeTensorDesc dgate_desc;
  constexpr int gate_num = 4;
  vector<int64_t> dgate_dims = {batch_size, gate_num * hidden_size};
  SetTensorDescription(dgate_desc, dgate_dims, ge::FORMAT_ND, state_type);
  basic_cell_desc->AddOutputDesc("dgate", dgate_desc);

  ge::GeTensorDesc dct_pre_desc;
  vector<int64_t> dct_dims = {batch_size, hidden_size};
  SetTensorDescription(dct_pre_desc, dct_dims, ge::FORMAT_ND, state_type);
  basic_cell_desc->AddOutputDesc("dct_1", dct_pre_desc);

  ge::AttrUtils::SetFloat(basic_cell_desc, "forget_bias", 0.0);
  ge::AttrUtils::SetStr(basic_cell_desc, "activation", "Tanh");
  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(fused_desc, "direction", direction);
  ge::AttrUtils::SetStr(basic_cell_desc, "direction", direction);

  std::string gate_order = "ijfo";  // DynamicLSTMGradCell only support ijfo
  ge::AttrUtils::GetStr(fused_desc, "gate_order", gate_order);
  ge::AttrUtils::SetStr(basic_cell_desc, "gate_order", gate_order);
  return basic_cell_desc;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicAddLSTMInputGradNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                                    vector<ge::NodePtr>& new_nodes) {
  std::string curr_name = grad_name + "/currT";
  ge::OpDescPtr const_t0_desc = CreateScaleConstDesc(curr_name, 0);
  ge::NodePtr const_t0_node = AddNewNode(graph, const_t0_desc, new_nodes);

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  ge::GeTensorDesc t0_output_desc = const_t0_desc->GetOutputDesc(0).Clone();
  ge::OpDescPtr cell_op_desc = GetDynamicLSTMGradCellDesc(fused_desc, t0_output_desc);
  ge::NodePtr cell_node = AddNewNode(graph, cell_op_desc, new_nodes);

  ge::GraphUtils::AddEdge(const_t0_node->GetOutDataAnchor(0),
                          cell_node->GetInDataAnchor(LSTM_GRAD_CELL_INPUT_INDEX["t_state"]));
  for (auto idx : LSTM_GRAD_CELL_INPUT_INDEX) {
    std::string key = idx.first;
    if (key == "t_state" || key == "mask") {
      continue;
    }
    ge::NodePtr unsqueeze_node =
        DynamicUnsqueezeNode(key, fused_desc->GetInputDesc(RNN_GRAD_INPUT_INDEX[key]), graph, new_nodes);
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX[key])->GetPeerOutAnchor(),
                            unsqueeze_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(unsqueeze_node->GetOutDataAnchor(0), cell_node->GetInDataAnchor(idx.second));
  }

  // add edge of output tensor dc0
  constexpr int cell_output_index_dc0 = 1;
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DC0)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(cell_node->GetOutDataAnchor(cell_output_index_dc0), in_anchor);
  }
  return cell_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDbReduceSumNode(const ge::NodePtr& t0_cell_node, ge::NodePtr& fused_node,
                                                               ge::ComputeGraph& graph,
                                                               vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/db/ReduceSum";
  // create reduce_sum desc
  ge::OpDescPtr sum_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((sum_desc = std::make_shared<ge::OpDesc>(node_name, "ReduceSumD")), return nullptr);

  // input tensor is dgate
  constexpr int gate_num = 4;
  ge::GeTensorDesc dgate_desc;
  vector<int64_t> dgate_dims = {batch_size, gate_num * hidden_size};
  SetTensorDescription(dgate_desc, dgate_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  sum_desc->AddInputDesc("x", dgate_desc);

  ge::GeTensorDesc output_desc;
  vector<int64_t> output_dims = {gate_num * hidden_size};
  SetTensorDescription(output_desc, output_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  sum_desc->AddOutputDesc("y", output_desc);

  // attr
  const vector<int64_t> axis = {0};
  ge::AttrUtils::SetListInt(sum_desc, "axes", axis);
  ge::AttrUtils::SetBool(sum_desc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr sum_node = AddNewNode(graph, sum_desc, new_nodes);
  ge::GraphUtils::AddEdge(t0_cell_node->GetOutDataAnchor(0), sum_node->GetInDataAnchor(0));

  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DB)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(sum_node->GetOutDataAnchor(0), in_anchor);
  }
  return sum_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicWConcatNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/w/ConcatD";
  ge::OpDescPtr concat_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((concat_desc = std::make_shared<ge::OpDesc>(node_name, "ConcatD")), return nullptr);

  constexpr int gate_num = 4;
  ge::GeTensorDesc wx_desc;
  vector<int64_t> wx_dims = {input_size, gate_num * hidden_size};
  SetTensorDescription(wx_desc, wx_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x0", wx_desc);

  ge::GeTensorDesc wh_desc;
  vector<int64_t> wh_dims = {hidden_size, gate_num * hidden_size};
  SetTensorDescription(wh_desc, wh_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x1", wh_desc);

  ge::GeTensorDesc output_desc;
  vector<int64_t> output_dims = {input_size + hidden_size, gate_num * hidden_size};
  SetTensorDescription(output_desc, output_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddOutputDesc("y", output_desc);

  constexpr int concat_axis = 0;
  ge::AttrUtils::SetInt(concat_desc, "concat_dim", concat_axis);
  constexpr int concat_num = 2;
  ge::AttrUtils::SetInt(concat_desc, "N", concat_num);

  ge::NodePtr concat_node = AddNewNode(graph, concat_desc, new_nodes);
  ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["wx"])->GetPeerOutAnchor(),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["wh"])->GetPeerOutAnchor(),
                          concat_node->GetInDataAnchor(1));
  return concat_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicXHConcatNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/xh/ConcatD";
  ge::OpDescPtr concat_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((concat_desc = std::make_shared<ge::OpDesc>(node_name, "ConcatD")), return nullptr);

  ge::GeTensorDesc x_desc;
  vector<int64_t> x_dims = {batch_size, input_size};
  SetTensorDescription(x_desc, x_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x0", x_desc);

  ge::GeTensorDesc h_desc;
  vector<int64_t> h_dims = {batch_size, hidden_size};
  SetTensorDescription(h_desc, h_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x1", h_desc);

  ge::GeTensorDesc output_desc;
  vector<int64_t> output_dims = {batch_size, input_size + hidden_size};
  SetTensorDescription(output_desc, output_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddOutputDesc("y", output_desc);

  constexpr int concat_axis = 1;
  ge::AttrUtils::SetInt(concat_desc, "concat_dim", concat_axis);
  constexpr int concat_num = 2;
  ge::AttrUtils::SetInt(concat_desc, "N", concat_num);

  ge::NodePtr concat_node = AddNewNode(graph, concat_desc, new_nodes);
  ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["h0"])->GetPeerOutAnchor(),
                          concat_node->GetInDataAnchor(1));
  return concat_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicXHSplitNode(const ge::NodePtr& dxh_matmul_node, ge::NodePtr& fused_node,
                                                           ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) {
  std::string node_name = grad_name + "/dxh/Split";
  ge::OpDescPtr split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((split_desc = std::make_shared<ge::OpDesc>(node_name, "SplitV")), return nullptr);

  ge::GeTensorDesc dxh_desc;
  vector<int64_t> dxh_dims = {batch_size, input_size + hidden_size};
  SetTensorDescription(dxh_desc, dxh_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddInputDesc("x", dxh_desc);

  ge::GeTensorDesc output1_desc;
  vector<int64_t> output1_dims = {batch_size, input_size};
  SetTensorDescription(output1_desc, output1_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dx", output1_desc);

  ge::GeTensorDesc output2_desc;
  vector<int64_t> output2_dims = {batch_size, hidden_size};
  SetTensorDescription(output2_desc, output2_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dh0", output2_desc);

  int32_t num_split = 2;
  std::vector<std::string> const_tensor_name = {"size_splits", "split_dim"};
  ge::GeTensorDesc size_splits_desc;
  vector<int64_t> size_splits_dims = {num_split};
  SetTensorDescription(size_splits_desc, size_splits_dims, ge::FORMAT_ND, ge::DT_INT64);
  split_desc->AddInputDesc(const_tensor_name[0], size_splits_desc);
  ge::GeTensorDesc split_dim_desc;
  vector<int64_t> split_dim = {1};
  SetTensorDescription(split_dim_desc, split_dim, ge::FORMAT_ND, ge::DT_INT32);
  split_desc->AddInputDesc(const_tensor_name[1], split_dim_desc);
  split_desc->SetOpInferDepends(const_tensor_name);

  ge::AttrUtils::SetInt(split_desc, "num_split", num_split);
  ge::NodePtr split_node = AddNewNode(graph, split_desc, new_nodes);

  // create const node and connect const node with split
  ge::Operator::OpListInt size_splits_value = {input_size, hidden_size};
  ge::NodePtr const_size_splits_node =
      CreateConstNode<int64_t>("/dxh/size_splits", size_splits_desc, size_splits_value, num_split, graph, new_nodes);

  ge::Operator::OpListInt split_dim_value = {1};
  ge::NodePtr const_dim_node =
      CreateConstNode<int32_t>("/dxh/split_dim", split_dim_desc, split_dim_value, 1, graph, new_nodes);

  ge::GraphUtils::AddEdge(dxh_matmul_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_size_splits_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const_dim_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(2));

  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DX)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), in_anchor);
  }
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DH0)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), in_anchor);
  }
  return split_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDwSplitNode(const ge::NodePtr& dw_matmul_node, ge::NodePtr& fused_node,
                                                           ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) {
  std::string node_name = grad_name + "/dw/Split";
  ge::OpDescPtr split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((split_desc = std::make_shared<ge::OpDesc>(node_name, "SplitVD")), return nullptr);

  constexpr int gate_num = 4;
  ge::GeTensorDesc dw_desc;
  vector<int64_t> dw_dims = {input_size + hidden_size, gate_num * hidden_size};
  SetTensorDescription(dw_desc, dw_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddInputDesc("dw", dw_desc);

  ge::GeTensorDesc output1_desc;
  vector<int64_t> output1_dims = {input_size, gate_num * hidden_size};
  SetTensorDescription(output1_desc, output1_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dwx", output1_desc);

  ge::GeTensorDesc output2_desc;
  vector<int64_t> output2_dims = {hidden_size, gate_num * hidden_size};
  SetTensorDescription(output2_desc, output2_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dwh", output2_desc);

  int32_t num_split = 2;
  std::vector<int64_t> size_splits_value = {input_size, hidden_size};
  ge::AttrUtils::SetListInt(split_desc, "size_splits", size_splits_value);
  int split_dim_value = 0;
  ge::AttrUtils::SetInt(split_desc, "split_dim", split_dim_value);
  ge::AttrUtils::SetInt(split_desc, "num_split", num_split);
  ge::NodePtr split_node = AddNewNode(graph, split_desc, new_nodes);

  ge::GraphUtils::AddEdge(dw_matmul_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DWX)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), in_anchor);
  }
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DWH)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), in_anchor);
  }
  return split_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicUnsqueezeNode(const std::string& name,
                                                             const ge::GeTensorDesc& input_desc,
                                                             ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) {
  auto unsqueeze_op = ge::OperatorFactory::CreateOperator(name.c_str(), "Unsqueeze");
  FUSION_PASS_CHECK(unsqueeze_op.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Unsqueeze Op operator error"),
                    return nullptr);
  auto unsqueeze_op_desc = ge::OpDescUtils::GetOpDescFromOperator(unsqueeze_op);
  unsqueeze_op.BreakConnect();

  vector<int64_t> output_dims = input_desc.GetShape().GetDims();
  output_dims.emplace(output_dims.begin(), 1);  // output dims is [1, input_dims]

  ge::GeTensorDesc output_desc;
  SetTensorDescription(output_desc, output_dims, ge::FORMAT_ND, input_desc.GetDataType());

  FUSION_PASS_CHECK(SUCCESS != unsqueeze_op_desc->UpdateInputDesc("x", input_desc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Unsqueeze node update inputDesc failed!"), return nullptr);
  FUSION_PASS_CHECK(SUCCESS != unsqueeze_op_desc->UpdateOutputDesc("y", output_desc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Unsqueeze node update outputDesc failed!"), return nullptr);

  constexpr int64_t axis = 0;
  ge::AttrUtils::SetListInt(unsqueeze_op_desc, "axes", {axis});
  ge::NodePtr unsqueeze_node = AddNewNode(graph, unsqueeze_op_desc, new_nodes);
  return unsqueeze_node;
}

void DynamicRNNV2GradFusionPass::GetNodeInfo(ge::NodePtr& fused_node) {
  ge::GeShape x_shape = fused_node->GetOpDesc()->GetInputDesc(RNN_GRAD_INPUT_INDEX["x"]).GetShape();
  constexpr int64_t x_index_batch = 0;
  constexpr int64_t x_index_input = 1;
  // shape of x is [batch_size, input_size]
  batch_size = x_shape.GetDim(x_index_batch);
  input_size = x_shape.GetDim(x_index_input);

  // shape of wh is [hidden_size, 4*hidden_size]
  constexpr int64_t wh_index_hidden = 0;
  hidden_size = fused_node->GetOpDesc()->GetInputDesc(RNN_GRAD_INPUT_INDEX["wh"]).GetShape().GetDim(wh_index_hidden);
  state_type = fused_node->GetOpDesc()->GetInputDesc(RNN_GRAD_INPUT_INDEX["i"]).GetDataType();
  grad_name = fused_node->GetName();
  return;
}

Status DynamicRNNV2GradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get dynamicRNNV2GradNode
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Get DynamicRNNV2Grad Node Failed."),
                    return FAILED);
  GetNodeInfo(fused_node);

  // add lstm cell grad, output is  dgate & dct_1
  ge::NodePtr t0_cell_node = DynamicAddLSTMInputGradNode(fused_node, graph, newNodes);
  // add concate(wx, wh)
  ge::NodePtr w_concat_node = DynamicWConcatNode(fused_node, graph, newNodes);
  // add matmul(dgate, w), output is dxh
  ge::NodePtr dxh_matmul_node = DynamicDxhMatMulNode(t0_cell_node, w_concat_node, graph, newNodes);
  // add concate(x, h0)
  ge::NodePtr xh_concat_node = DynamicXHConcatNode(fused_node, graph, newNodes);
  // add matmul(xy, dgate), output is dw
  ge::NodePtr dw_matmul_node = DynamicDwMatMulNode(xh_concat_node, t0_cell_node, graph, newNodes);
  // add reducesum(dgate), output is db
  ge::NodePtr db_sum_node = DynamicDbReduceSumNode(t0_cell_node, fused_node, graph, newNodes);
  // add split(dw), output is dwx & dwh
  ge::NodePtr dw_split_node = DynamicDwSplitNode(dw_matmul_node, fused_node, graph, newNodes);
  // add split(dxh), output is dx & dh
  ge::NodePtr dxh_split_node = DynamicXHSplitNode(dxh_matmul_node, fused_node, graph, newNodes);

  for (auto inAnchor : fused_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove fused_node from graph
  if (graph.RemoveNode(fused_node) != ge::GRAPH_SUCCESS) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", grad_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_PASS("DynamicRNNV2GradFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNV2GradFusionPass);
}  // namespace fe
