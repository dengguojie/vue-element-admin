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
const int FRACTAL_SHAPE = 16;
const int GRAD_OUT_IDX_DWX = 0;
const int GRAD_OUT_IDX_DWH = 1;
const int GRAD_OUT_IDX_DB = 2;
const int GRAD_OUT_IDX_DX = 3;
const int GRAD_OUT_IDX_DH0 = 4;
const int GRAD_OUT_IDX_DC0 = 5;

map<std::string, int> RNN_GRAD_INPUT_INDEX = {
    {"x", 0},   {"wx", 1},      {"wh", 2},       {"h0", 4},   {"c0", 5},   {"dy", 8},   {"dh", 9},
    {"dc", 10}, {"y", 3},       {"h", 6},        {"c", 7},    {"i", 11},   {"j", 12},   {"f", 13},
    {"o", 14},  {"tanhct", 15}, {"seq_len", 16}, {"wci", 17}, {"wcf", 18}, {"wco", 19}, {"mask", 20}};
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

void DynamicRNNV2GradFusionPass::MakUpRange(ge::GeTensorDesc& desc, const vector<int64_t>& dims) const {
  constexpr int64_t unknown_shape = -1;
  auto found = std::find(dims.begin(), dims.end(), unknown_shape);
  if (found == dims.end()) {
    return;
  }

  vector<std::pair<int64_t, int64_t>> range;
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

ge::GeTensorDesc DynamicRNNV2GradFusionPass::SetTensorDescription(const vector<int64_t>& dims,
                                                                  const ge::Format& format,
                                                                  const ge::DataType& dtype) const {
  ge::GeTensorDesc desc;
  ge::GeShape shape(dims);
  desc.SetShape(shape);
  desc.SetDataType(dtype);
  desc.SetFormat(format);
  desc.SetOriginShape(shape);
  desc.SetOriginDataType(dtype);
  desc.SetOriginFormat(format);
  MakUpRange(desc, dims);
  return desc;
}

ge::GeTensorDesc DynamicRNNV2GradFusionPass::SetTensorDescription(const vector<int64_t>& dims,
                                                                  const ge::Format& format, const ge::DataType& dtype,
                                                                  const vector<int64_t>& ori_dims,
                                                                  const ge::Format& ori_format) const {
  ge::GeTensorDesc desc;
  ge::GeShape shape(dims);
  desc.SetShape(shape);
  desc.SetDataType(dtype);
  desc.SetFormat(format);
  ge::GeShape ori_shape(ori_dims);
  desc.SetOriginShape(ori_shape);
  desc.SetOriginDataType(dtype);
  desc.SetOriginFormat(ori_format);
  MakUpRange(desc, dims);
  return desc;
}

ge::NodePtr DynamicRNNV2GradFusionPass::AddNewNode(ge::ComputeGraph& graph, const ge::OpDescPtr& op_desc,
                                                   vector<ge::NodePtr>& new_nodes) const {
  ge::NodePtr node = graph.AddNode(op_desc);
  if (!node) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add new node failed.");
    return nullptr;
  }
  new_nodes.push_back(node);
  return node;
}

ge::OpDescPtr DynamicRNNV2GradFusionPass::CreateScaleConstDesc(const std::string& name, int32_t value) const {
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
                                                        vector<T>& const_data, ge::ComputeGraph& graph,
                                                        vector<ge::NodePtr>& new_nodes) const {
  ge::OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>(name, "Const");
  if (const_op_desc == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_op_desc is nullptr.");
    return nullptr;
  }
  int32_t length = const_data.size();
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

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDxhMatMulNode(const ge::NodePtr& t0_cell_node,
                                                             const ge::NodePtr& w_concat_node, ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dxh/MatMul";
  ge::OpDescPtr matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmul_desc = std::make_shared<ge::OpDesc>(node_name, "BatchMatMulV2")), return nullptr);
  // left tensor is dgate
  constexpr int gate_num = 4;
  vector<int64_t> dgate_nd_dims = {batch_size, gate_num * hidden_nz_size * FRACTAL_SHAPE};
  vector<int64_t> dgate_nz_dims = {gate_num * hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  ge::GeTensorDesc dgate_desc =
      SetTensorDescription(dgate_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, dgate_nd_dims, ge::FORMAT_ND);
  matmul_desc->AddInputDesc("x1", dgate_desc);
  // right tensor is w.T
  vector<int64_t> w_nd_dims = {input_size + hidden_size, gate_num * hidden_size};
  vector<int64_t> w_nz_dims = {input_nz_size + hidden_nz_size, gate_num * hidden_nz_size, FRACTAL_SHAPE,
                               FRACTAL_SHAPE};
  ge::GeTensorDesc w_desc =
      SetTensorDescription(w_nz_dims, ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16, w_nd_dims, ge::FORMAT_ND);
  matmul_desc->AddInputDesc("x2", w_desc);
  // add matmul output
  vector<int64_t> dxy_nd_dims = {batch_size, (input_nz_size + hidden_nz_size) * FRACTAL_SHAPE};
  vector<int64_t> dxy_nz_dims = {input_nz_size + hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  ge::GeTensorDesc dxy_desc =
      SetTensorDescription(dxy_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, dxy_nd_dims, ge::FORMAT_ND);
  matmul_desc->AddOutputDesc("y", dxy_desc);
  // attr
  AttrUtils::SetBool(matmul_desc, "adj_x1", false);
  AttrUtils::SetBool(matmul_desc, "adj_x2", true);
  AttrUtils::SetInt(matmul_desc, "input_size", input_size);
  AttrUtils::SetInt(matmul_desc, "hidden_size", hidden_size);
  ge::NodePtr matmul_node = AddNewNode(graph, matmul_desc, new_nodes);
  ge::GraphUtils::AddEdge(t0_cell_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(w_concat_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
  return matmul_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDwMatMulNode(const ge::NodePtr& xh_node,
                                                            const ge::NodePtr& t0_cell_node, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dw/MatMul";
  ge::OpDescPtr matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmul_desc = std::make_shared<ge::OpDesc>(node_name, "BatchMatMul")), return nullptr);
  // left tensor is xh.T
  vector<int64_t> xh_dims = {batch_size, (input_nz_size + hidden_nz_size) * FRACTAL_SHAPE};
  ge::GeTensorDesc xh_desc = SetTensorDescription(xh_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x1", xh_desc);
  // left tensor is dgate
  constexpr int gate_num = 4;
  vector<int64_t> dgate_dims = {batch_size, gate_num * hidden_nz_size * FRACTAL_SHAPE};
  ge::GeTensorDesc dgate_desc = SetTensorDescription(dgate_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddInputDesc("x2", dgate_desc);
  // add matmul output
  vector<int64_t> dw_dims = {(input_nz_size + hidden_nz_size) * FRACTAL_SHAPE,
                             gate_num * hidden_nz_size * FRACTAL_SHAPE};
  ge::GeTensorDesc dw_tensor_desc = SetTensorDescription(dw_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul_desc->AddOutputDesc("y", dw_tensor_desc);
  // attr
  ge::AttrUtils::SetBool(matmul_desc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmul_desc, "adj_x2", false);
  ge::NodePtr matmul_node = AddNewNode(graph, matmul_desc, new_nodes);
  ge::GraphUtils::AddEdge(xh_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(t0_cell_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
  return matmul_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::AddDwTrasposeNode(const ge::NodePtr& dw_node, ge::ComputeGraph& graph,
                                                          vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dw/Transpose";
  // create transpose desc
  ge::OpDescPtr transpose_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((transpose_desc = std::make_shared<ge::OpDesc>(node_name, "TransposeD")), return nullptr);
  // input for transpose
  constexpr int gate_num = 4;
  vector<int64_t> src_nd_dims = {(input_nz_size + hidden_nz_size) * FRACTAL_SHAPE,
                                 hidden_nz_size * gate_num * FRACTAL_SHAPE};
  vector<int64_t> src_nz_dims = {hidden_nz_size * gate_num, input_nz_size + hidden_nz_size, FRACTAL_SHAPE,
                                 FRACTAL_SHAPE};
  ge::GeTensorDesc src_tensor_desc =
      SetTensorDescription(src_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, src_nd_dims, ge::FORMAT_ND);
  transpose_desc->AddInputDesc("x", src_tensor_desc);
  // output for transpose
  vector<int64_t> dsc_nd_dims = {input_size + hidden_size, hidden_size * gate_num};
  vector<int64_t> dsc_nz_dims = {input_nz_size + hidden_nz_size, hidden_nz_size * gate_num, FRACTAL_SHAPE,
                                 FRACTAL_SHAPE};
  ge::GeTensorDesc dsc_tensor_desc =
      SetTensorDescription(dsc_nz_dims, ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16, dsc_nd_dims, ge::FORMAT_ND);
  transpose_desc->AddOutputDesc("trans_dsc", dsc_tensor_desc);
  // attr
  vector<int32_t> perm_value = {1, 0, 3, 2};  // transpose from Nz to Zn
  ge::AttrUtils::SetListInt(transpose_desc, "perm", perm_value);
  ge::AttrUtils::SetInt(transpose_desc, "input_size", input_size);
  ge::AttrUtils::SetInt(transpose_desc, "hidden_size", hidden_size);

  // create transpose node
  ge::NodePtr transpose_node = AddNewNode(graph, transpose_desc, new_nodes);
  ge::GraphUtils::AddEdge(dw_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  return transpose_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::AddDwTrasDataNode(const ge::NodePtr& transpose_node, ge::ComputeGraph& graph,
                                                          vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dw/TransDataRNN";
  // create transdata desc
  ge::OpDescPtr transdata_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdata_desc = std::make_shared<ge::OpDesc>(node_name, "TransDataRNN")), return nullptr);
  // input for transdata
  constexpr int gate_num = 4;
  vector<int64_t> w_nd_dims = {input_size + hidden_size, hidden_size * gate_num};
  vector<int64_t> w_rnn_dims = {input_nz_size + hidden_nz_size, hidden_nz_size * gate_num, FRACTAL_SHAPE,
                                FRACTAL_SHAPE};

  ge::GeTensorDesc src_tensor_desc =
      SetTensorDescription(w_rnn_dims, ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16, w_nd_dims, ge::FORMAT_ND);
  transdata_desc->AddInputDesc("trans_src", src_tensor_desc);
  // output for tarnsdata

  ge::GeTensorDesc dsc_tensor_desc = SetTensorDescription(w_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  transdata_desc->AddOutputDesc("trans_dsc", dsc_tensor_desc);
  // attr
  AttrUtils::SetStr(transdata_desc, "src_format", "FRACTAL_ZN_RNN");
  AttrUtils::SetStr(transdata_desc, "dst_format", "ND");
  AttrUtils::SetInt(transdata_desc, "input_size", input_size);
  AttrUtils::SetInt(transdata_desc, "hidden_size", hidden_size);

  // create transpose node
  ge::NodePtr transdata_rnn_node = AddNewNode(graph, transdata_desc, new_nodes);
  ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), transdata_rnn_node->GetInDataAnchor(0));
  return transdata_rnn_node;
}

ge::OpDescPtr DynamicRNNV2GradFusionPass::GetDynamicLSTMGradCellDesc(ge::OpDescPtr& fused_desc,
                                                                     ge::GeTensorDesc& curT_desc) const {
  std::string node_name = grad_name + "/DynamicLSTMGradCell0";
  ge::OpDescPtr basic_cell_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((basic_cell_desc = std::make_shared<ge::OpDesc>(node_name, "DynamicLSTMGradCell")),
                          return nullptr);

  constexpr int64_t time_step = 1;
  vector<int64_t> input_nz_dims = {time_step, hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  vector<int64_t> input_dims = {time_step, batch_size, hidden_size};
  ge::GeTensorDesc input_desc =
      SetTensorDescription(input_nz_dims, ge::FORMAT_FRACTAL_NZ, state_type, input_dims, ge::FORMAT_ND);

  constexpr int cell_index_time = 10;
  for (int i = 0; i < cell_index_time; i++) {
    basic_cell_desc->AddInputDesc(i, input_desc);
  }
  basic_cell_desc->AddInputDesc(cell_index_time, curT_desc);

  constexpr int gate_num = 4;
  vector<int64_t> dgate_nz_dims = {hidden_nz_size * gate_num, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  vector<int64_t> dgate_nd_dims = {batch_size, hidden_nz_size * gate_num * FRACTAL_SHAPE};
  ge::GeTensorDesc dgate_desc =
      SetTensorDescription(dgate_nz_dims, ge::FORMAT_FRACTAL_NZ, state_type, dgate_nd_dims, ge::FORMAT_ND);
  basic_cell_desc->AddOutputDesc("dgate", dgate_desc);

  vector<int64_t> dct_nz_dims = {hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  vector<int64_t> dct_nd_dims = {batch_size, hidden_size};
  ge::GeTensorDesc dct_pre_desc =
      SetTensorDescription(dct_nz_dims, ge::FORMAT_FRACTAL_NZ, state_type, dct_nd_dims, ge::FORMAT_ND);
  basic_cell_desc->AddOutputDesc("dct_1", dct_pre_desc);

  ge::AttrUtils::SetFloat(basic_cell_desc, "forget_bias", 0.0);
  ge::AttrUtils::SetStr(basic_cell_desc, "activation", "Tanh");
  std::string direction = "UNIDIRECTIONAL";
  ge::AttrUtils::GetStr(fused_desc, "direction", direction);
  ge::AttrUtils::SetStr(basic_cell_desc, "direction", direction);

  std::string gate_order = "ijfo";
  ge::AttrUtils::GetStr(fused_desc, "gate_order", gate_order);
  ge::AttrUtils::SetStr(basic_cell_desc, "gate_order", gate_order);
  return basic_cell_desc;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicAddLSTMInputGradNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                                    vector<ge::NodePtr>& new_nodes) const {
  map<std::string, int> cell_2d_input_index = {{"c0", 0}, {"dh", 3}, {"dc", 4}};
  map<std::string, int> cell_3d_input_index = {{"c", 1}, {"dy", 2}, {"i", 5},     {"j", 6},
                                               {"f", 7}, {"o", 8},  {"tanhct", 9}};
  // create t0 const data
  std::string curr_name = grad_name + "/currT";
  ge::OpDescPtr const_t0_desc = CreateScaleConstDesc(curr_name, 0);
  ge::NodePtr const_t0_node = AddNewNode(graph, const_t0_desc, new_nodes);
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  ge::GeTensorDesc t0_output_desc = const_t0_desc->GetOutputDesc(0).Clone();
  ge::OpDescPtr cell_op_desc = GetDynamicLSTMGradCellDesc(fused_desc, t0_output_desc);
  ge::NodePtr cell_node = AddNewNode(graph, cell_op_desc, new_nodes);
  ge::GraphUtils::AddEdge(const_t0_node->GetOutDataAnchor(0), cell_node->GetInDataAnchor(10));  // idx of t_state is 10

  // connect 2d-input, with unsqueeze
  for (auto idx : cell_2d_input_index) {
    std::string key = idx.first;
    ge::NodePtr unsqueeze_node =
        DynamicUnsqueezeNode(key, fused_desc->GetInputDesc(RNN_GRAD_INPUT_INDEX[key]), graph, new_nodes);
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX[key])->GetPeerOutAnchor(),
                            unsqueeze_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(unsqueeze_node->GetOutDataAnchor(0), cell_node->GetInDataAnchor(idx.second));
  }
  for (auto idx : cell_3d_input_index) {
    std::string key = idx.first;
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX[key])->GetPeerOutAnchor(),
                            cell_node->GetInDataAnchor(idx.second));
  }

  // add edge of output tensor dc0
  constexpr int cell_output_index_dc0 = 1;
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DC0)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(cell_node->GetOutDataAnchor(cell_output_index_dc0), in_anchor);
  }
  return cell_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DbTransDataNode(const ge::NodePtr& sum_node, ge::ComputeGraph& graph,
                                                        vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/db/TransDataRNN";
  // create transdata desc
  ge::OpDescPtr transdata_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdata_desc = std::make_shared<ge::OpDesc>(node_name, "TransDataRNN")), return nullptr);
  // input for transdata
  constexpr int gate_num = 4;
  vector<int64_t> db_nd_dims = {hidden_size * gate_num};
  vector<int64_t> db_rnn_dims = {hidden_nz_size * gate_num * FRACTAL_SHAPE};

  ge::GeTensorDesc src_tensor_desc =
      SetTensorDescription(db_rnn_dims, ge::FORMAT_ND_RNN_BIAS, ge::DT_FLOAT16, db_nd_dims, ge::FORMAT_ND);
  transdata_desc->AddInputDesc("trans_src", src_tensor_desc);
  // output for tarnsdata
  ge::GeTensorDesc dsc_tensor_desc = SetTensorDescription(db_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  transdata_desc->AddOutputDesc("trans_dsc", dsc_tensor_desc);
  // attr
  AttrUtils::SetStr(transdata_desc, "src_format", "ND_RNN_BIAS");
  AttrUtils::SetStr(transdata_desc, "dst_format", "ND");
  AttrUtils::SetInt(transdata_desc, "input_size", input_size);
  AttrUtils::SetInt(transdata_desc, "hidden_size", hidden_size);

  // create transpose node
  ge::NodePtr transdata_rnn_node = AddNewNode(graph, transdata_desc, new_nodes);
  if (transdata_rnn_node == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add new node of transdata rnn failed.");
    return nullptr;
  }
  ge::GraphUtils::AddEdge(sum_node->GetOutDataAnchor(0), transdata_rnn_node->GetInDataAnchor(0));
  return transdata_rnn_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDbReduceSumNode(const ge::NodePtr& t0_cell_node,
                                                               ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                               vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/db/ReduceSum";
  // create reduce_sum desc
  ge::OpDescPtr sum_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((sum_desc = std::make_shared<ge::OpDesc>(node_name, "ReduceSumD")), return nullptr);

  // input tensor is dgate
  constexpr int gate_num = 4;
  vector<int64_t> dgate_nd_dims = {batch_size, hidden_nz_size * gate_num * FRACTAL_SHAPE};
  vector<int64_t> dgate_ori_dims = {batch_size, hidden_size * gate_num};
  ge::GeTensorDesc dgate_desc =
      SetTensorDescription(dgate_nd_dims, ge::FORMAT_ND, state_type, dgate_ori_dims, ge::FORMAT_ND);
  sum_desc->AddInputDesc("x", dgate_desc);

  vector<int64_t> output_ori_dims = {gate_num * hidden_size};
  vector<int64_t> output_nd_dims = {gate_num * hidden_nz_size * FRACTAL_SHAPE};
  ge::GeTensorDesc output_desc =
      SetTensorDescription(output_nd_dims, ge::FORMAT_ND_RNN_BIAS, state_type, output_ori_dims, ge::FORMAT_ND);
  sum_desc->AddOutputDesc("y", output_desc);
  // attr
  const vector<int64_t> axis = {0};
  ge::AttrUtils::SetListInt(sum_desc, "axes", axis);
  ge::AttrUtils::SetBool(sum_desc, "keep_dims", false);
  ge::AttrUtils::SetInt(sum_desc, "hidden_size", hidden_size);

  // create reduce_sum node
  ge::NodePtr sum_node = AddNewNode(graph, sum_desc, new_nodes);
  if (sum_node == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add reducesum for db failed.");
    return nullptr;
  }
  ge::GraphUtils::AddEdge(t0_cell_node->GetOutDataAnchor(0), sum_node->GetInDataAnchor(0));
  ge::NodePtr trans_node = DbTransDataNode(sum_node, graph, new_nodes);
  if (trans_node == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add transdata rnn for db failed.");
    return nullptr;
  }
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DB)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(trans_node->GetOutDataAnchor(0), in_anchor);
  }
  return sum_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicWConcatNode(ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/w/ConcatD";
  ge::OpDescPtr concat_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((concat_desc = std::make_shared<ge::OpDesc>(node_name, "ConcatD")), return nullptr);

  constexpr int gate_num = 4;
  vector<int64_t> wx_dims = {input_size, gate_num * hidden_size};
  ge::GeTensorDesc wx_desc = SetTensorDescription(wx_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x0", wx_desc);

  vector<int64_t> wh_dims = {hidden_size, gate_num * hidden_size};
  ge::GeTensorDesc wh_desc = SetTensorDescription(wh_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  concat_desc->AddInputDesc("x1", wh_desc);

  vector<int64_t> output_dims = {input_size + hidden_size, gate_num * hidden_size};
  ge::GeTensorDesc output_desc = SetTensorDescription(output_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
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

  vector<int64_t> x_dims = {batch_size, input_nz_size * FRACTAL_SHAPE};
  vector<int64_t> x_nz_dims = {input_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  ge::GeTensorDesc x_desc =
      SetTensorDescription(x_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, x_dims, ge::FORMAT_ND);
  concat_desc->AddInputDesc("x0", x_desc);

  vector<int64_t> h_dims = {batch_size, hidden_nz_size * FRACTAL_SHAPE};
  vector<int64_t> h_nz_dims = {hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  ge::GeTensorDesc h_desc =
      SetTensorDescription(h_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, h_dims, ge::FORMAT_ND);
  concat_desc->AddInputDesc("x1", h_desc);

  vector<int64_t> output_dims = {batch_size, (input_nz_size + hidden_nz_size) * FRACTAL_SHAPE};
  vector<int64_t> output_nz_dims = {input_nz_size + hidden_nz_size, batch_nz_size, FRACTAL_SHAPE, FRACTAL_SHAPE};
  ge::GeTensorDesc output_desc =
      SetTensorDescription(output_nz_dims, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, output_dims, ge::FORMAT_ND);
  concat_desc->AddOutputDesc("y", output_desc);

  constexpr int concat_axis = 1;
  ge::AttrUtils::SetInt(concat_desc, "concat_dim", concat_axis);
  constexpr int concat_num = 2;
  ge::AttrUtils::SetInt(concat_desc, "N", concat_num);
  ge::NodePtr concat_node = AddNewNode(graph, concat_desc, new_nodes);

  if (is_input_size_aligned) {
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["x"])->GetPeerOutAnchor(),
                            concat_node->GetInDataAnchor(0));
  } else {
    vector<int64_t> x_ori_dims = {batch_size, input_size};
    std::string x_pad_name = grad_name + "/x_pad";
    ge::NodePtr x_pad_node = CreatePadNode(x_pad_name, x_ori_dims, x_dims, graph, new_nodes);
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["x"])->GetPeerOutAnchor(),
                            x_pad_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(x_pad_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  }

  if (is_hidden_size_aligned) {
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["h0"])->GetPeerOutAnchor(),
                            concat_node->GetInDataAnchor(1));
  } else {
    vector<int64_t> h_ori_dims = {batch_size, hidden_size};
    std::string h_pad_name = grad_name + "/h0_pad";
    ge::NodePtr h_pad_node = CreatePadNode(h_pad_name, h_ori_dims, h_dims, graph, new_nodes);
    ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(RNN_GRAD_INPUT_INDEX["h0"])->GetPeerOutAnchor(),
                            h_pad_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(h_pad_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  }
  return concat_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::CreatePadNode(std::string& name, vector<int64_t>& src_dims,
                                                      vector<int64_t>& dsc_dims, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& new_nodes) const {
  ge::OpDescPtr pad_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((pad_desc = std::make_shared<ge::OpDesc>(name, "Pad")), return nullptr);

  ge::GeTensorDesc x_desc = SetTensorDescription(src_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  pad_desc->AddInputDesc("x", x_desc);
  // input2 padding

  vector<int64_t> pad_input2_dims = {2, 2};  // size of src_dims is 2
  ge::GeTensorDesc paddings_desc = SetTensorDescription(pad_input2_dims, ge::FORMAT_ND, ge::DT_INT32);
  pad_desc->AddInputDesc("paddings", paddings_desc);

  ge::GeTensorDesc output_desc = SetTensorDescription(dsc_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  pad_desc->AddOutputDesc("y", output_desc);
  vector<string> depend_names = {"paddings"};
  pad_desc->SetOpInferDepends(depend_names);
  ge::NodePtr pad_node = AddNewNode(graph, pad_desc, new_nodes);

  // create const node and connect const node with pad
  vector<int32_t> paddings_value = {0, 0, 0, static_cast<int32_t>(dsc_dims[1] - src_dims[1])};
  ge::NodePtr const_padding_node =
      CreateConstNode<int32_t>(name + "/paddings", paddings_desc, paddings_value, graph, new_nodes);
  ge::GraphUtils::AddEdge(const_padding_node->GetOutDataAnchor(0), pad_node->GetInDataAnchor(1));
  return pad_node;
}

vector<int64_t> DynamicRNNV2GradFusionPass::CreateSplitTensorDesc(ge::OpDescPtr& split_desc) const {
  vector<int64_t> dxh_nd_dims = {batch_size, (input_nz_size + hidden_nz_size) * FRACTAL_SHAPE};
  ge::GeTensorDesc dxh_desc = SetTensorDescription(dxh_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddInputDesc("x", dxh_desc);

  vector<int64_t> dx_nd_dims = {batch_size, input_size};
  ge::GeTensorDesc dx_desc = SetTensorDescription(dx_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dx", dx_desc);
  vector<int64_t> size_splits_value = {input_size};

  if (!is_input_size_aligned) {
    int64_t input_tail_size = input_nz_size * FRACTAL_SHAPE - input_size;
    vector<int64_t> dx_tail_nd_dims = {batch_size, input_tail_size};
    ge::GeTensorDesc dx_tail_desc = SetTensorDescription(dx_tail_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
    split_desc->AddOutputDesc("dx_tail", dx_tail_desc);
    size_splits_value.push_back(input_tail_size);
  }

  vector<int64_t> dh_nd_dims = {batch_size, hidden_size};
  ge::GeTensorDesc dh_desc = SetTensorDescription(dh_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dh", dh_desc);
  size_splits_value.push_back(hidden_size);

  if (!is_hidden_size_aligned) {
    int64_t hidden_tail_size = hidden_nz_size * FRACTAL_SHAPE - hidden_size;
    vector<int64_t> dh_tail_nd_dims = {batch_size, hidden_tail_size};
    ge::GeTensorDesc dh_tail_desc = SetTensorDescription(dh_tail_nd_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
    split_desc->AddOutputDesc("dh_tail", dh_tail_desc);
    size_splits_value.push_back(hidden_tail_size);
  }
  return size_splits_value;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicXHSplitNode(const ge::NodePtr& dxh_matmul_node, ge::NodePtr& fused_node,
                                                           ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dxh/Split";
  ge::OpDescPtr split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((split_desc = std::make_shared<ge::OpDesc>(node_name, "SplitV")), return nullptr);
  vector<int64_t> size_splits_value=CreateSplitTensorDesc(split_desc);
  vector<std::string> const_tensor_name = {"size_splits", "split_dim"};
  int64_t num_split = size_splits_value.size();
  vector<int64_t> size_splits_dims = {num_split};
  ge::GeTensorDesc size_splits_desc = SetTensorDescription(size_splits_dims, ge::FORMAT_ND, ge::DT_INT64);
  split_desc->AddInputDesc(const_tensor_name[0], size_splits_desc);
  vector<int64_t> split_dim = {1};
  ge::GeTensorDesc split_dim_desc = SetTensorDescription(split_dim, ge::FORMAT_ND, ge::DT_INT32);
  split_desc->AddInputDesc(const_tensor_name[1], split_dim_desc);
  split_desc->SetOpInferDepends(const_tensor_name);
  ge::AttrUtils::SetInt(split_desc, "num_split", num_split);
  ge::NodePtr split_node = AddNewNode(graph, split_desc, new_nodes);

  // create const node and connect const node with split
  ge::NodePtr const_size_splits_node =
      CreateConstNode<int64_t>("/dxh/size_splits", size_splits_desc, size_splits_value, graph, new_nodes);
  vector<int32_t> split_dim_value = {1};  // split axis is 1
  ge::NodePtr const_dim_node =
      CreateConstNode<int32_t>("/dxh/split_dim", split_dim_desc, split_dim_value, graph, new_nodes);

  ge::GraphUtils::AddEdge(dxh_matmul_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_size_splits_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const_dim_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(2));
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DX)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), in_anchor);
  }
  int dh_output_idx = is_input_size_aligned ? 1 : 2;  // if input size is aligned, dh_idx is 1. otherwise, idx is 2.
  for (auto in_anchor : fused_node->GetOutDataAnchor(GRAD_OUT_IDX_DH0)->GetPeerInDataAnchors()) {
    in_anchor->UnlinkAll();
    ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(dh_output_idx), in_anchor);
  }
  return split_node;
}

ge::NodePtr DynamicRNNV2GradFusionPass::DynamicDwSplitNode(const ge::NodePtr& dw_matmul_node, ge::NodePtr& fused_node,
                                                           ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& new_nodes) const {
  std::string node_name = grad_name + "/dw/Split";
  ge::OpDescPtr split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((split_desc = std::make_shared<ge::OpDesc>(node_name, "SplitVD")), return nullptr);

  constexpr int gate_num = 4;
  vector<int64_t> dw_dims = {input_size + hidden_size, gate_num * hidden_size};
  ge::GeTensorDesc dw_desc = SetTensorDescription(dw_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddInputDesc("dw", dw_desc);

  vector<int64_t> output1_dims = {input_size, gate_num * hidden_size};
  ge::GeTensorDesc output1_desc = SetTensorDescription(output1_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dwx", output1_desc);

  vector<int64_t> output2_dims = {hidden_size, gate_num * hidden_size};
  ge::GeTensorDesc output2_desc = SetTensorDescription(output2_dims, ge::FORMAT_ND, ge::DT_FLOAT16);
  split_desc->AddOutputDesc("dwh", output2_desc);

  int32_t num_split = 2;
  vector<int64_t> size_splits_value = {input_size, hidden_size};
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
                                                             ge::ComputeGraph& graph,
                                                             vector<ge::NodePtr>& new_nodes) const {
  auto unsqueeze_op = ge::OperatorFactory::CreateOperator(name.c_str(), "Unsqueeze");
  FUSION_PASS_CHECK(unsqueeze_op.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Unsqueeze Op operator error"),
                    return nullptr);
  auto unsqueeze_op_desc = ge::OpDescUtils::GetOpDescFromOperator(unsqueeze_op);
  unsqueeze_op.BreakConnect();

  vector<int64_t> output_dims = input_desc.GetShape().GetDims();
  output_dims.emplace(output_dims.begin(), 1);  // output dims is [1, input_dims]

  ge::GeTensorDesc output_desc = SetTensorDescription(output_dims, ge::FORMAT_ND, input_desc.GetDataType());
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
  batch_nz_size = (batch_size + FRACTAL_SHAPE - 1) / FRACTAL_SHAPE;
  input_size = x_shape.GetDim(x_index_input);
  input_nz_size = (input_size + FRACTAL_SHAPE - 1) / FRACTAL_SHAPE;

  // shape of wh is [hidden_size, 4*hidden_size]
  constexpr int64_t wh_index_hidden = 0;
  hidden_size = fused_node->GetOpDesc()->GetInputDesc(RNN_GRAD_INPUT_INDEX["wh"]).GetShape().GetDim(wh_index_hidden);
  hidden_nz_size = (hidden_size + FRACTAL_SHAPE - 1) / FRACTAL_SHAPE;
  state_type = fused_node->GetOpDesc()->GetInputDesc(RNN_GRAD_INPUT_INDEX["i"]).GetDataType();
  grad_name = fused_node->GetName();

  is_input_size_aligned = (input_size % FRACTAL_SHAPE == 0);
  is_hidden_size_aligned = (hidden_size % FRACTAL_SHAPE == 0);
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
  // transpose dw from Nz to Zn
  ge::NodePtr dw_transpose_node = AddDwTrasposeNode(dw_matmul_node, graph, newNodes);
  // transdata dw from Zn to nd
  ge::NodePtr dw_transdata_node = AddDwTrasDataNode(dw_transpose_node, graph, newNodes);
  // add split(dw), output is dwx & dwh
  ge::NodePtr dw_split_node = DynamicDwSplitNode(dw_transdata_node, fused_node, graph, newNodes);
  // add reducesum(dgate), output is db
  ge::NodePtr db_sum_node = DynamicDbReduceSumNode(t0_cell_node, fused_node, graph, newNodes);
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
