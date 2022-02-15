/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 LSTMP   ->   DATA_WX   DATA_WR
                  \       /
                   \     /
                   Concat    DATA_PROJECT
                      |           |
                      |           |
                   Transpose    Transpose
                       \          /
                        \        /
                      DynamicRNNV3
                            |
                            |
                          Slice
*/

#include "lstmp_fusion_pass.h"

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

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "LSTMP";
static const std::string PATTERN_FUSEDNODE = "LSTMP";
static const int32_t SPLIT_LEN = 2;
static const int32_t SPLIT_DIM = 3;
static const int32_t LAST_DIM = 2;
static const int32_t J_OFFSET = 2;
static const int32_t ROW_NUM = 4;
static const int32_t FORMAT_ALIGN = 16;
static const int32_t WR_IDX = 0;
static const int32_t WC_IDX = 1;
static const int32_t RR_IDX = 2;
static const int32_t RC_IDX = 3;
static const int32_t I_IDX = 2;

static map<std::string, int> LSTMP_INPUT_INDEX = {
    {"x", 0}, {"wx", 1}, {"bias", 2}, {"wr", 3}, {"project", 4}, {"real_mask", 5},
    {"init_h", 6}, {"init_c", 7}};
static map<std::string, int> OUTPUT_INDEX = {{"y", 0}, {"output_h", 1}, {"output_c", 2}};

static map<std::string, int> V3_INPUT_INDEX = {
    {"x", 0}, {"w", 1}, {"b", 2}, {"init_h", 3}, {"init_c", 4}, {"project", 5}};

vector<FusionPattern*> LSTMPFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("LSTMPFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status LSTMPFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion start.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null."),
                    return PARAM_INVALID);

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."),
                    return PARAM_INVALID);

  ge::NodePtr dynamicv3_node = nullptr;
  FUSION_PASS_CHECK(CreateDynamicV3Node(graph, fused_desc, fused_node, dynamicv3_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateDynamicV3Node FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(AddEdgeForInput(graph, fused_node, dynamicv3_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdgeForInput FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(AddEdgeForOutput(graph, fused_node, dynamicv3_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdgeForOutput FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(RemoveFusedNode(graph, fused_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RemoveFusedNode FAIL"),
                    return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion success.");
  return SUCCESS;
}

Status LSTMPFusionPass::CreateTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& input_desc,
                                            ge::NodePtr& new_node, std::vector<int32_t>& perm,
                                            const std::string& name) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateTransposeNode start.");
  std::shared_ptr<ge::OpDesc> trans_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (trans_desc = std::make_shared<ge::OpDesc>(name + "_transpose", "Transpose")),
    return FAILED);

  FUSION_PASS_CHECK(trans_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create Transpose node name %s",
                                                   name.c_str()),
                    return FAILED);

  auto input_dims = input_desc.GetShape().GetDims();
  int64_t dim_num = input_desc.GetShape().GetDimNum();
  FUSION_PASS_CHECK(dim_num != perm.size(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "input dims size[%ld] should be equal perm size[%ld]",
                                                   dim_num, perm.size()),
                    return FAILED);

  std::vector<int64_t> new_dims;
  for (std::size_t i = 0; i < dim_num; ++i) {
    new_dims.push_back(input_dims[perm[i]]);
  }

  std::vector<int64_t> perm_dims = {dim_num};
  ge::GeShape shape(perm_dims);
  ge::GeTensorDesc perm_desc(shape, ge::FORMAT_ND, DT_INT32);

  auto output_desc = input_desc.Clone();
  output_desc.SetShape(ge::GeShape(new_dims));
  output_desc.SetOriginShape(ge::GeShape(new_dims));

  trans_desc->AddInputDesc("x", input_desc);
  trans_desc->AddInputDesc("perm", perm_desc);
  trans_desc->AddOutputDesc("y", output_desc);

  vector<string> permVec;
  permVec.push_back("perm");
  trans_desc->SetOpInferDepends(permVec);

  new_node = graph.AddNode(trans_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create Transpose node"),
                    return FAILED);

  std::shared_ptr<ge::OpDesc> perm_const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (perm_const_desc = std::make_shared<ge::OpDesc>(name + "_perm_const", "Const")),
    return FAILED);

  FUSION_PASS_CHECK(perm_const_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);

  ge::GeTensorPtr tensor_ptr = nullptr;

  FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                              perm_desc, reinterpret_cast<uint8_t*>(perm.data()),
                              dim_num * sizeof(int32_t))),
                          return PARAM_INVALID);

  ge::AttrUtils::SetTensor(perm_const_desc, ge::ATTR_NAME_WEIGHTS, tensor_ptr);
  perm_const_desc->AddOutputDesc(tensor_ptr->GetTensorDesc());
  auto perm_node = graph.AddNode(perm_const_desc);
  FUSION_PASS_CHECK(perm_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to CreateConstNode node"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(perm_node->GetOutDataAnchor(0), new_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add perm edge to transpose failed."),
        return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateTransposeNode success.");
  return SUCCESS;
}

Status LSTMPFusionPass::CreateDynamicV3Node(ge::ComputeGraph& graph, const ge::OpDescPtr& fused_desc,
                                            const ge::NodePtr& fused_node, ge::NodePtr& new_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateDynamicV3Node start.");
  std::shared_ptr<ge::OpDesc> dynamic_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (dynamic_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_dynamicV3", "DynamicRNNV3")),
    return FAILED);

  FUSION_PASS_CHECK(dynamic_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create DynamicRNNV3 node"),
                    return FAILED);

  auto input_wp = fused_desc->GetInputDesc("project").Clone();
  auto dims_wp = input_wp.GetShape().GetDims();

  FUSION_PASS_CHECK(dims_wp.size() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "input wp dim_num should be 2, cur is[%ld]", dims_wp.size()),
                    return FAILED);
  int state = dims_wp[0];
  int hidden_size = dims_wp[1];
  auto x_desc = fused_desc->GetInputDesc("x").Clone();
  auto dims_x = x_desc.GetShape().GetDims();

  int batch_size = dims_x[0];
  int num_step = dims_x[1];
  int input_size = dims_x[I_IDX];

  ge::AttrUtils::SetInt(dynamic_desc, "input_size", input_size);
  ge::AttrUtils::SetInt(dynamic_desc, "hidden_size", hidden_size);
  ge::AttrUtils::SetInt(dynamic_desc, "state_size", state);

  std::vector<int64_t> dims_x_trans = {num_step, batch_size, input_size};
  x_desc.SetShape(ge::GeShape(dims_x_trans));
  x_desc.SetOriginShape(ge::GeShape(dims_x_trans));
  dynamic_desc->AddInputDesc("x", x_desc);

  std::vector<ge::GeTensorPtr> wTensorPtrs;
  Status retW = ProcessLSTMWxr(fused_node, wTensorPtrs);
  FUSION_PASS_CHECK(retW != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
  dynamic_desc->AddInputDesc("w", wTensorPtrs[0]->GetTensorDesc());

  std::vector<ge::GeTensorPtr> bTensorPtrs;
  Status retB = ProcessLSTMb(fused_node, bTensorPtrs);
  FUSION_PASS_CHECK(retB != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process b fail."), return FAILED);
  dynamic_desc->AddInputDesc("b", bTensorPtrs[0]->GetTensorDesc());

  if (fused_desc->MutableInputDesc("init_h") != nullptr) {
      auto input_desc = fused_desc->GetInputDesc("init_h");
      dynamic_desc->AddInputDesc("init_h", input_desc);
  }

  if (fused_desc->MutableInputDesc("init_c") != nullptr) {
      auto input_desc = fused_desc->GetInputDesc("init_c");
      dynamic_desc->AddInputDesc("init_c", input_desc);
  }

  std::vector<int64_t> dims_project = {hidden_size, state};
  input_wp.SetShape(ge::GeShape(dims_project));
  input_wp.SetOriginShape(ge::GeShape(dims_project));
  dynamic_desc->AddInputDesc("project", input_wp);

  auto outputx_desc = x_desc.Clone();
  auto outputh_desc = outputx_desc.Clone();
  auto outputc_desc = outputx_desc.Clone();

  std::vector<int64_t> dims_y = {num_step, batch_size, state};
  outputx_desc.SetShape(ge::GeShape(dims_y));
  outputx_desc.SetOriginShape(ge::GeShape(dims_y));
  dynamic_desc->AddOutputDesc("y", outputx_desc);

  outputh_desc.SetShape(ge::GeShape(dims_y));
  outputh_desc.SetOriginShape(ge::GeShape(dims_y));
  dynamic_desc->AddOutputDesc("output_h", outputh_desc);

  std::vector<int64_t> dims_c = {num_step, batch_size, hidden_size};
  outputc_desc.SetShape(ge::GeShape(dims_c));
  outputc_desc.SetOriginShape(ge::GeShape(dims_c));
  dynamic_desc->AddOutputDesc("output_c", outputc_desc);

  std::vector<std::string> out_names = {"i", "j", "f", "o", "tanhc"};
  for (auto& name : out_names) {
    dynamic_desc->AddOutputDesc(name, outputc_desc);
  }

  new_node = graph.AddNode(dynamic_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to CreateDynamicV3Node node"),
                    return FAILED);

  // connect w
  ge::OpDescPtr wDescForward = ge::OpDescUtils::CreateConstOp(wTensorPtrs[0]);
  ge::NodePtr wForward_node = graph.AddNode(wDescForward);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(wForward_node->GetOutDataAnchor(0),
                    new_node->GetInDataAnchor(V3_INPUT_INDEX["w"])),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add forward dynamicRnn edge to fusion node w failed."),
  return FAILED);

  // connect bias
  ge::OpDescPtr bDescForward = ge::OpDescUtils::CreateConstOp(bTensorPtrs[0]);
  ge::NodePtr bForward_node = graph.AddNode(bDescForward);
  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(bForward_node->GetOutDataAnchor(0),
                                           new_node->GetInDataAnchor(V3_INPUT_INDEX["b"])),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias input edge to fusion node  failed."),
        return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateDynamicV3Node success.");
  return SUCCESS;
}

void LSTMPFusionPass::SetTensorDescription(ge::GeTensorDesc &tensorDesc, const vector<int64_t> &dims,
                                           const ge::Format &format, const ge::DataType &dtype) {
    ge::GeShape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(format);
    tensorDesc.SetOriginShape(shape);
    tensorDesc.SetOriginDataType(dtype);
    tensorDesc.SetOriginFormat(format);
    return;
}

template <class T>
static Status SetWeightTensorData(ge::GeTensorPtr wTensorPtr, ge::GeTensorPtr rTensorPtr,
                                  std::vector<int32_t> &inputDims, ge::GeTensorDesc tensorDesc,
                                  std::vector<int32_t> &start_size, vector<ge::GeTensorPtr> &tensorPtr) {
  int32_t wRow = inputDims[WR_IDX];
  int32_t wCol = inputDims[WC_IDX];
  int32_t rRow = inputDims[RR_IDX];
  int32_t rCol = inputDims[RC_IDX];
  int32_t targetCol = wCol + rCol;
  int32_t weightSize = targetCol * wRow;

  // the wx + wr matrix
  unique_ptr<T[]> wxrMergeData(new (std::nothrow) T[weightSize]());
  FUSION_PASS_CHECK(wxrMergeData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxrMergeData is NULL"),
                    return FAILED);
  T *wxData = (T *)wTensorPtr->GetData().data();
  T *wrData = (T *)rTensorPtr->GetData().data();
  FUSION_PASS_CHECK(wxData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxData is NULL"), return FAILED);
  FUSION_PASS_CHECK(wrData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wrData is NULL"), return FAILED);

  auto retMem = memset_s(wxrMergeData.get(), weightSize, 0, weightSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "Failed to operate memset_s function!"),
                    return FAILED);

  int32_t wx_start_size = start_size[0];
  int32_t wr_start_size = start_size[1];

  // wx transpose, assign to merge data
  T *dstWeight = wxrMergeData.get();
  for (int32_t i = 0; i < wRow * wCol; ++i) {
    *(dstWeight + i / wCol + wRow * (i % wCol)) = *(wxData + i + wx_start_size);
  }

  // wr transpose, assign to merge data
  for (int32_t i = 0; i < rRow * rCol; ++i) {
    *(dstWeight + wRow * wCol + i / rCol + rRow * (i % rCol)) = *(wrData + i + wr_start_size);
  }

  // func: ifjo -> ijfo
  int32_t beginSize = wRow / ROW_NUM;
  for (int32_t col = 0; col < targetCol; ++col) {
    for (int32_t row = 0; row < beginSize; ++row) {
      T tmp = *(dstWeight + col * wRow + beginSize + row);
      dstWeight[col * wRow + beginSize + row] = dstWeight[col * wRow + beginSize * J_OFFSET + row];
      dstWeight[col * wRow + beginSize * J_OFFSET + row] = tmp;
    }
  }

  ge::GeTensorPtr dstTensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((dstTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(wxrMergeData.get()), weightSize * sizeof(T))),
                           return FAILED);
  tensorPtr.push_back(dstTensorPtr);

  return SUCCESS;
}

Status LSTMPFusionPass::ProcessLSTMWxr(const ge::NodePtr fused_node, vector<ge::GeTensorPtr> &tensorPtr) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "has enter process onnx lstm W");
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["wx"]);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constWxAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsW = ge::OpDescUtils::MutableWeights(inputWNode);
  FUSION_PASS_CHECK(weightsW.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null."),
                    return FAILED);
  ge::GeTensorPtr wTensorPtr = weightsW[0];

  ge::InDataAnchorPtr inputRAnchorPtr0 = fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["wr"]);
  ge::OutDataAnchorPtr constRAnchorPtr0 = inputRAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputRNode = constRAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsR = ge::OpDescUtils::MutableWeights(inputRNode);
  FUSION_PASS_CHECK(weightsR.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsR is null."),
                    return FAILED);
  ge::GeTensorPtr rTensorPtr = weightsR[0];

  ge::GeTensorDesc wConstTensorDesc = wTensorPtr->GetTensorDesc();
  ge::GeTensorDesc rConstTensorDesc = rTensorPtr->GetTensorDesc();

  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  int32_t wRow = wConstTensorDesc.GetShape().GetDim(0);
  int32_t wCol = wConstTensorDesc.GetShape().GetDim(1);
  int32_t rRow = rConstTensorDesc.GetShape().GetDim(0);
  int32_t rCol = rConstTensorDesc.GetShape().GetDim(1);
  FUSION_PASS_CHECK(wCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "wCol can not 0"), return FAILED);
  FUSION_PASS_CHECK(rCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "rCol can not 0"), return FAILED);

  ge::GeTensorDesc weightTensorDesc;
  std::vector<int64_t> dimsIn = {wCol + rCol, wRow};
  SetTensorDescription(weightTensorDesc, dimsIn, ge::FORMAT_ND, dataType);

  std::vector<int32_t> inputDims{wRow, wCol, rRow, rCol};
  Status ret = SUCCESS;

  std::vector<int32_t> start_size = {0, 0};
  if (dataType == ge::DT_FLOAT16) {
    ret = SetWeightTensorData<uint16_t>(wTensorPtr, rTensorPtr, inputDims,
                                        weightTensorDesc, start_size, tensorPtr);
  } else if (dataType == ge::DT_FLOAT) {
    ret = SetWeightTensorData<float>(wTensorPtr, rTensorPtr, inputDims,
                                     weightTensorDesc, start_size, tensorPtr);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32).",
                                   fused_node->GetName().c_str());
    return FAILED;
  }

  return ret;
}

template <class T>
static Status SetBiasTensorData(ge::GeTensorPtr bTensorPtr, int32_t biasLen,
                                ge::GeTensorDesc tensorDesc, vector<ge::GeTensorPtr> &tensorPtr) {
  // the wx + wr matrix
  unique_ptr<T[]> bData(new (std::nothrow) T[biasLen]());
  FUSION_PASS_CHECK(bData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "bData is NULL"),
                    return FAILED);
  T *oriBData = (T *)bTensorPtr->GetData().data();
  FUSION_PASS_CHECK(oriBData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "oriBData is NULL"),
                    return FAILED);

  auto retMem = memset_s(bData.get(), biasLen, 0, biasLen);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "Failed to operate memset_s function!"),
                    return FAILED);

  T *dstWeight = bData.get();
  for (int32_t i = 0; i < biasLen; ++i) {
    *(dstWeight + i) = *(oriBData + i);
  }
  int32_t beginSize = biasLen / ROW_NUM;
  // func: ifjo -> ijfo
  for (int32_t row = 0; row < beginSize; ++row) {
    T tmp = *(dstWeight + beginSize + row);
    dstWeight[beginSize + row] = dstWeight[beginSize * J_OFFSET + row];
    dstWeight[beginSize * J_OFFSET + row] = tmp;
  }

  ge::GeTensorPtr dstTensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((dstTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(bData.get()), biasLen * sizeof(T))),
                           return FAILED);
  tensorPtr.push_back(dstTensorPtr);

  return SUCCESS;
}

Status LSTMPFusionPass::ProcessLSTMb(const ge::NodePtr fused_node, vector<ge::GeTensorPtr> &tensorPtr) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ProcessLSTMb start");

  ge::InDataAnchorPtr inputBiasAnchorPtr0 = fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["bias"]);
  ge::OutDataAnchorPtr constBiasAnchorPtr0 = inputBiasAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputBNode = constBiasAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> Bias = ge::OpDescUtils::MutableWeights(inputBNode);
  FUSION_PASS_CHECK(Bias.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null."),
                    return FAILED);
  ge::GeTensorPtr bTensorPtr = Bias[0];

  ge::GeTensorDesc bConstTensorDesc = bTensorPtr->GetTensorDesc();

  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  int32_t biasLen = bConstTensorDesc.GetShape().GetDim(0);

  FUSION_PASS_CHECK(biasLen == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bias len can not 0"),
                    return FAILED);

  ge::GeTensorDesc biasTensorDesc;
  std::vector<int64_t> dimsIn = {biasLen};
  SetTensorDescription(biasTensorDesc, dimsIn, ge::FORMAT_ND, dataType);

  Status ret = SUCCESS;

  std::vector<int32_t> start_size = {0, 0};
  if (dataType == ge::DT_FLOAT16) {
    ret = SetBiasTensorData<uint16_t>(bTensorPtr, biasLen, biasTensorDesc, tensorPtr);
  } else if (dataType == ge::DT_FLOAT) {
    ret = SetBiasTensorData<float>(bTensorPtr, biasLen, biasTensorDesc, tensorPtr);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32).",
                                   fused_node->GetName().c_str());
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ProcessLSTMb success.");
  return ret;
}

Status LSTMPFusionPass::AddEdgeForInput(ge::ComputeGraph& graph, const ge::NodePtr& fused_node,
                                        ge::NodePtr& dynamicv3_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddEdgeForInput");
  auto fused_desc = fused_node->GetOpDesc();
  auto input_x_desc = fused_desc->GetInputDesc("x");
  ge::NodePtr trans_x = nullptr;
  std::vector<int32_t> perm_x = {1, 0, 2};
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_x_desc, trans_x, perm_x, "x") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "X CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["x"])->GetPeerOutAnchor(),
                                           trans_x->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x input edge to trans_x failed."),
        return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_x->GetOutDataAnchor(0),
                                           dynamicv3_node->GetInDataAnchor(V3_INPUT_INDEX["x"])),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add trans_x edge to fusion node failed."),
        return FAILED);

  int32_t idx_offset = 0;
  if (fused_desc->MutableInputDesc("init_h") != nullptr) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                                           dynamicv3_node->GetInDataAnchor(V3_INPUT_INDEX["init_h"])),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h input edge to fusion node  failed."),
        return FAILED);
  } else {
    ++idx_offset;
  }

  if (fused_desc->MutableInputDesc("init_c") != nullptr) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["init_c"])->GetPeerOutAnchor(),
                                           dynamicv3_node->GetInDataAnchor(V3_INPUT_INDEX["init_c"] - idx_offset)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_c input edge to fusion node  failed."),
        return FAILED);
  } else {
    ++idx_offset;
  }

  ge::NodePtr trans_wp = nullptr;
  std::vector<int32_t> perm_p = {1, 0};
  auto input_desc = fused_desc->GetInputDesc("project");
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_desc, trans_wp, perm_p, "project") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(
          fused_node->GetInDataAnchor(LSTMP_INPUT_INDEX["project"])->GetPeerOutAnchor(),
          trans_wp->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add project input edge to trans_wp failed."),
        return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_wp->GetOutDataAnchor(0),
                                           dynamicv3_node->GetInDataAnchor(V3_INPUT_INDEX["project"] - idx_offset)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const project input edge to fusion node failed."),
        return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddEdgeForInput success.");
  return SUCCESS;
}

Status LSTMPFusionPass::AddEdgeForOutput(ge::ComputeGraph& graph, const ge::NodePtr& fused_node,
                                         ge::NodePtr& dynamicv3_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddEdgeForOutput");

  auto dynamicv3_desc = dynamicv3_node->GetOpDesc();
  auto input_y_desc = dynamicv3_desc->GetOutputDesc("y");
  std::vector<int32_t> perm = {1, 0, 2};
  ge::NodePtr trans_y = nullptr;
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_y_desc, trans_y, perm, "y") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Y CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(0), trans_y->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add y output edge to trans_y failed."),
        return FAILED);

  for (auto in_anchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for y fail"),
          return FAILED);
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_y->GetOutDataAnchor(0), in_anchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add trans_y edge to fusion node failed."),
        return FAILED);
  }

  auto output_desc = dynamicv3_desc->GetOutputDesc("output_h");
  auto dims = output_desc.GetShape().GetDims();
  int32_t num_size = dims[0];

  if (num_size > 1) {
    ge::NodePtr splith_node = nullptr;

    FUSION_PASS_CHECK(CreateSplitNode(graph, dynamicv3_desc, splith_node, "output_h") != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateSplitNode output_h FAIL"),
                      return FAILED);

    FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_h"]),
                                            splith_node->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splith node  failed."),
          return FAILED);

    for (auto in_anchor : fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_h"])->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(1), in_anchor),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for init_h fail"),
            return FAILED);
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(splith_node->GetOutDataAnchor(1), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add splith edge to fusion node failed."),
          return FAILED);
    }

    ge::NodePtr splitc_node = nullptr;

    FUSION_PASS_CHECK(CreateSplitNode(graph, dynamicv3_desc, splitc_node, "output_c") != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateSplitNode output_c FAIL"),
                      return FAILED);

    FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"]),
                                            splitc_node->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splitc node  failed."),
          return FAILED);

    for (auto in_anchor : fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"])->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"]), in_anchor),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for init_c fail"),
            return FAILED);
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(splitc_node->GetOutDataAnchor(1), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add splitc edge to fusion node failed."),
          return FAILED);
    }
  } else {
    for (auto in_anchor : fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_h"])->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(1), in_anchor),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for init_h fail"),
            return FAILED);
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_h"]), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add splith edge to fusion node failed."),
          return FAILED);
    }

    for (auto in_anchor : fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"])->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"]), in_anchor),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for init_c fail"),
            return FAILED);
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"]), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add splitc edge to fusion node failed."),
          return FAILED);
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddEdgeForOutput success.");
  return SUCCESS;
}

Status LSTMPFusionPass::CreateSplitNode(ge::ComputeGraph& graph, const ge::OpDescPtr& dynamicv3_desc,
                                        ge::NodePtr& new_node, const std::string& name) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateSplitNode start.");
  auto input_desc = dynamicv3_desc->GetOutputDesc(name);
  std::shared_ptr<ge::OpDesc> split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (split_desc = std::make_shared<ge::OpDesc>(name + "_split", "SplitV")),
    return FAILED);

  FUSION_PASS_CHECK(split_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create split node"),
                    return FAILED);
  split_desc->AddInputDesc("x", input_desc);
  auto output_desc = input_desc.Clone();
  auto output_desc1 = input_desc.Clone();
  auto dims = output_desc.GetShape().GetDims();
  if (dims.size() != SPLIT_DIM) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dim size should be 3,cur is %ld", dims.size());
    return FAILED;
  }

  int32_t num_size = dims[0];
  int32_t batch_size = dims[1];
  int32_t last_size = dims[LAST_DIM];

  std::vector<int64_t> new_dims = {num_size - 1, batch_size, last_size};
  output_desc.SetShape(ge::GeShape(new_dims));
  output_desc.SetOriginShape(ge::GeShape(new_dims));
  split_desc->AddOutputDesc("y0", output_desc);

  std::vector<int64_t> new_dims1 = {1, batch_size, last_size};
  output_desc1.SetShape(ge::GeShape(new_dims1));
  output_desc1.SetOriginShape(ge::GeShape(new_dims1));
  split_desc->AddOutputDesc("y1", output_desc1);

  ge::Operator::OpListInt size_splits = {num_size - 1, 1};
  ge::Operator::OpListInt split_dim = {0};

  std::vector<int64_t> size_splits_dims = {2};
  ge::GeShape size_splits_shape(size_splits_dims);
  ge::GeTensorDesc size_splits_desc(size_splits_shape, ge::FORMAT_ND, DT_INT64);

  split_desc->AddInputDesc("size_splits", size_splits_desc);

  std::vector<int64_t> split_dims = {1};
 
  ge::GeShape split_dims_shape(split_dims);
  ge::GeTensorDesc split_dim_desc(split_dims_shape, ge::FORMAT_ND, DT_INT32);

  split_desc->AddInputDesc("split_dim", split_dim_desc);

  ge::AttrUtils::SetInt(split_desc, "num_split", SPLIT_LEN);

  vector<string> constVec = {"size_splits", "split_dim"};
  split_desc->SetOpInferDepends(constVec);

  new_node = graph.AddNode(split_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateSplitNode node"),
                    return FAILED);

  std::shared_ptr<ge::OpDesc> size_splits_const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (size_splits_const_desc = std::make_shared<ge::OpDesc>(name + "_size_splits_const", "Const")),
    return FAILED);

  FUSION_PASS_CHECK(size_splits_const_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);

  ge::GeTensorPtr size_splits_tensor_ptr = nullptr;

  FUSION_PASS_MAKE_SHARED((size_splits_tensor_ptr = std::make_shared<ge::GeTensor>(
                              size_splits_desc, reinterpret_cast<uint8_t*>(size_splits.data()),
                              2 * sizeof(int64_t))),
                          return PARAM_INVALID);

  ge::AttrUtils::SetTensor(size_splits_const_desc, ge::ATTR_NAME_WEIGHTS, size_splits_tensor_ptr);
  size_splits_const_desc->AddOutputDesc(size_splits_tensor_ptr->GetTensorDesc());
  auto size_splits_node = graph.AddNode(size_splits_const_desc);
  FUSION_PASS_CHECK(size_splits_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to CreateConstNode node"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(size_splits_node->GetOutDataAnchor(0),
                                           new_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add size_splits_node edge to split failed."),
        return FAILED);

  std::shared_ptr<ge::OpDesc> split_dim_const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (split_dim_const_desc = std::make_shared<ge::OpDesc>(name + "_split_dim_const", "Const")),
    return FAILED);

  FUSION_PASS_CHECK(split_dim_const_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);

  ge::GeTensorPtr tensor_ptr = nullptr;

  FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                              split_dim_desc, reinterpret_cast<uint8_t*>(split_dim.data()),
                              sizeof(int32_t))),
                          return PARAM_INVALID);

  ge::AttrUtils::SetTensor(split_dim_const_desc, ge::ATTR_NAME_WEIGHTS, tensor_ptr);
  split_dim_const_desc->AddOutputDesc(tensor_ptr->GetTensorDesc());
  auto split_dim_node = graph.AddNode(split_dim_const_desc);
  FUSION_PASS_CHECK(split_dim_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to CreateConstNode node"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(split_dim_node->GetOutDataAnchor(0),
                                           new_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add split_dim_node edge to split failed."),
        return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateSplitNode success.");
  return SUCCESS;
}

Status LSTMPFusionPass::RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node) {
  // unlink all control input of LSTMP
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveFusedNode start.");
  if (fused_node->GetInControlAnchor() != nullptr) {
    fused_node->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of LSTMP
  for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  // unlink all output of LSTMP
  for (auto out_Anchor : fused_node->GetAllOutDataAnchors()) {
    if (out_Anchor != nullptr) {
      out_Anchor->UnlinkAll();
    }
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                    fused_node->GetName().c_str()),
                    return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveFusedNode success.");
  return SUCCESS;
}
REGISTER_PASS("LSTMPFusionPass", BUILT_IN_GRAPH_PASS, LSTMPFusionPass);
} // fe