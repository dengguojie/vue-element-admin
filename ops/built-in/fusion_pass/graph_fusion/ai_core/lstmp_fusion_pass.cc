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
 LSTMP   ->    DATA_WX  DATA_WR
                  \       /
                   \     /
                    \   /
                    Concat    DATA_PROJECT
                      |           |
                   Transpose    Transpose
                       \          /
                        \        /
                        DynamicRNNV3
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

static map<std::string, int> INPUT_INDEX = {
    {"x", 0}, {"wx", 1}, {"bias", 2}, {"wr", 3}, {"project", 4}, {"real_mask", 5},
    {"init_h", 6}, {"init_c", 7}};
static map<std::string, int> OUTPUT_INDEX = {{"y", 0}, {"output_h", 1}, {"output_c", 2}};

vector<FusionPattern*> LSTMPFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("LSTMPFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status LSTMPFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {

  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
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
  return SUCCESS;
}

Status LSTMPFusionPass::CreateConstNode(ge::ComputeGraph&graph, ge::OpDescPtr& fused_desc, ge::NodePtr& new_node) {
  std::shared_ptr<ge::OpDesc> const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (const_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_const", "Const")),
    return FAILED);
  
  FUSION_PASS_CHECK(const_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);
  
  int len = fused_desc->GetInputDesc("x").GetShape().GetDims().at(1);
  DataType dataType = fused_desc->GetInputDesc("x").GetDataType();
  std::vector<int64_t>dims = {len};
  ge::GeShape shape(dims);
  ge::GeTensorDesc desc(shape, ge::FORMAT_ND, dataType);
  ge::GeTensorPtr tensor_ptr = nullptr;
  if (dataType == ge::DT_FLOAT16) {
    std::vector<uint16_t> val(len, 1);
    FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                               desc, reinterpret_cast<uint8_t*>(val.data()), len * sizeof(uint16_t))),
                            tensor_ptr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT) {
    std::vector<float> val(len, 1);
    FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                               desc, reinterpret_cast<uint8_t*>(val.data()), len * sizeof(float))),
                            tensor_ptr = nullptr;
                            return PARAM_INVALID);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "x's dtype is not in (float16, float32), fusion failed.");
    return FAILED;
  }

  ge::AttrUtils::SetTensor(const_desc, "value", tensor_ptr);
  const_desc->AddOutputDesc(tensor_ptr->GetTensorDesc());
  new_node = graph.AddNode(const_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateConstNode node"),
                    return FAILED);
  return SUCCESS;
}

Status LSTMPFusionPass::CreateTransposeNode(ge::ComputeGraph& graph, ge::GeTensorDesc& input_desc, ge::NodePtr& new_node, std::vector<int32_t>& perm,
                                            const std::string& name) {

  std::shared_ptr<ge::OpDesc> trans_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (trans_desc = std::make_shared<ge::OpDesc>(name + "_transpose", "TransposeD")),
    return FAILED);

  FUSION_PASS_CHECK(trans_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create Transpose node name %s", name.c_str()),
                    return FAILED);
  
  auto input_dims = input_desc.GetShape().GetDims();
  auto dim_num = input_desc.GetShape().GetDimNum();
  FUSION_PASS_CHECK(dim_num != perm.size(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input dims size[%d] should be equal perm size[%d]", (int)dim_num, (int)perm.size()),
                    return FAILED);
  
  std::vector<int64_t> new_dims;
  for (int i = 0; i < dim_num; ++i) {
    if (perm[i] >= dim_num) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "perm [%d] should less dim_num[%d]", (int)perm[i], (int)dim_num);
      return FAILED;
    }
    new_dims.push_back(input_dims[perm[i]]);
  }
  auto output_desc = input_desc.Clone();
  output_desc.SetShape(ge::GeShape(new_dims));
  output_desc.SetOriginShape(ge::GeShape(new_dims));
  trans_desc->AddInputDesc("x", input_desc);
  trans_desc->AddOutputDesc("y", output_desc);
  ge::AttrUtils::SetListInt(trans_desc, "perm", perm);
  new_node = graph.AddNode(trans_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create ConcatD node"),
                    return FAILED);
  return SUCCESS;
}

Status LSTMPFusionPass::CreateDynamicV3Node(ge::ComputeGraph& graph, ge::OpDescPtr& fused_desc, ge::NodePtr& fused_node,
                                            ge::NodePtr& new_node) {

  std::shared_ptr<ge::OpDesc> dynamic_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (dynamic_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_dynamicV3", "DynamicRNNV3")),
    return FAILED);

  FUSION_PASS_CHECK(dynamic_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create DynamicRNNV3 node"),
                    return FAILED);

  auto x_desc = fused_desc->GetInputDesc("x").Clone();
  auto dims_x = x_desc.GetShape().GetDims();
  std::vector<int64_t> dims_x_trans = {dims_x[1], dims_x[0], dims_x[2]};
  x_desc.SetShape(ge::GeShape(dims_x_trans));
  x_desc.SetOriginShape(ge::GeShape(dims_x_trans));
  dynamic_desc->AddInputDesc("x", x_desc);

  std::vector<ge::GeTensorPtr> wTensorPtrs;
  Status retW = ProcessLSTMWxh(fused_node, wTensorPtrs);
  FUSION_PASS_CHECK(retW != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
  dynamic_desc->AddInputDesc("w", wTensorPtrs[0]->GetTensorDesc());

  auto input_b_desc = fused_desc->GetInputDesc("bias");
  dynamic_desc->AddInputDesc("bias", input_b_desc);

  auto input_wp = fused_desc->GetInputDesc("project").Clone();
  auto dims_wp = input_wp.GetShape().GetDims();
  FUSION_PASS_CHECK(dims_wp.size() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input wp dim_num should be 2, cur is[%d]", (int)dims_wp.size()),
                    return FAILED);
  std::vector<int64_t> dims_project = {dims_wp[1], dims_wp[0]};
  input_wp.SetShape(ge::GeShape(dims_project));
  input_wp.SetOriginShape(ge::GeShape(dims_project));
  dynamic_desc->AddInputDesc("project", input_wp);

  auto input_mask_desc = fused_desc->GetInputDesc("x").Clone();
  std::vector<int64_t> dims = {x_desc.GetShape().GetDims().at(1)};
  input_mask_desc.SetShape(ge::GeShape(dims));
  input_mask_desc.SetOriginShape(ge::GeShape(dims));
  dynamic_desc->AddInputDesc("real_mask", input_mask_desc);

  std::vector<std::string> vec_input = {"init_h", "init_c"};
  for (auto& key : vec_input) {
    if (fused_desc->MutableInputDesc(key) != nullptr) {
      auto input_desc = fused_desc->GetInputDesc(key);
      dynamic_desc->AddInputDesc(key, input_desc);
    }
  }

  auto outputx_desc = x_desc.Clone();
  auto outputh_desc = outputx_desc.Clone();
  auto outputc_desc = outputx_desc.Clone();

  int seq = dims_x_trans[0];
  int batch = dims_x_trans[1];
  auto dims_wr = fused_desc->GetInputDesc("wr").GetShape().GetDims();
  int hidden = dims_wr[0] / 4;
  int state = dims_wr[1];

  std::vector<int64_t> dims_y = {seq, batch, state};
  outputx_desc.SetShape(ge::GeShape(dims_y));
  outputx_desc.SetOriginShape(ge::GeShape(dims_y));
  dynamic_desc->AddOutputDesc("y", outputx_desc);

  outputh_desc.SetShape(ge::GeShape(dims_y));
  outputh_desc.SetOriginShape(ge::GeShape(dims_y));
  dynamic_desc->AddOutputDesc("output_h", outputh_desc);

  std::vector<int64_t> dims_c = {seq, batch, hidden};
  outputc_desc.SetShape(ge::GeShape(dims_c));
  outputc_desc.SetOriginShape(ge::GeShape(dims_c));
  dynamic_desc->AddOutputDesc("output_c", outputc_desc);

  std::vector<std::string> out_names = {"i", "j", "f", "o", "tanhc"};
  for (auto& name : out_names) {
    dynamic_desc->AddOutputDesc(name, outputc_desc);
  }

  new_node = graph.AddNode(dynamic_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateDynamicV3Node node"),
                    return FAILED);

  //connect w
  ge::OpDescPtr wDescForward = ge::OpDescUtils::CreateConstOp(wTensorPtrs[0]);
  ge::NodePtr wForward_node = graph.AddNode(wDescForward);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(wForward_node->GetOutDataAnchor(0), new_node->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node w failed."),
  return FAILED);

  return SUCCESS;
}

void LSTMPFusionPass::SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims,
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
  int32_t wRow = inputDims[0];
  int32_t wCol = inputDims[1];
  int32_t rRow = inputDims[2];
  int32_t rCol = inputDims[3];
  int32_t targetCol = wCol + rCol;
  int32_t weightSize = targetCol * wRow;

  // the wx + wh matrix
  unique_ptr<T[]> wxhMergeData(new (std::nothrow) T[weightSize]());
  FUSION_PASS_CHECK(wxhMergeData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxhMergeData is NULL"), return FAILED);
  T *wxData = (T *)wTensorPtr->GetData().data();
  T *whData = (T *)rTensorPtr->GetData().data();
  FUSION_PASS_CHECK(wxData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxData is NULL"), return FAILED);
  FUSION_PASS_CHECK(whData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "whData is NULL"), return FAILED);

  auto retMem = memset_s(wxhMergeData.get(), weightSize, 0, weightSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "Failed to operate memset_s function!"), return FAILED);
  
  int32_t wx_start_size = start_size[0];
  int32_t wh_start_size = start_size[1];

  // wx transpose, assign to merge data
  T *dstWeight = wxhMergeData.get();
  for (int32_t i = 0; i < wRow * wCol; ++i) {
    *(dstWeight + i / wCol + wRow * (i % wCol)) = *(wxData + i + wx_start_size);
  }

  // wh transpose, assign to merge data
  for (int32_t i = 0; i < rRow * rCol; ++i) {
    *(dstWeight + wRow * wCol + i / rCol + rRow * (i % rCol)) = *(whData + i + wh_start_size);
  }

  // swap 1, 3
  int32_t beginSize = wRow / 4;
  for (int32_t col = 0; col < targetCol; ++col) {
    for (int32_t row = 0; row < beginSize; ++row) {
      T tmp = *(dstWeight + col * wRow + beginSize * 1 + row);
      dstWeight[col * wRow + beginSize * 1 + row] = dstWeight[col * wRow + beginSize * 3 + row];
      dstWeight[col * wRow + beginSize * 3 + row] = tmp;
    }
  }
  ge::GeTensorPtr dstTensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((dstTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(wxhMergeData.get()), weightSize * sizeof(T))),
                           return FAILED);
  tensorPtr.push_back(dstTensorPtr);

  return SUCCESS;
}

Status LSTMPFusionPass::ProcessLSTMWxh(ge::NodePtr fused_node, vector<ge::GeTensorPtr> &tensorPtr) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "has enter process onnx lstm W");
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fused_node->GetInDataAnchor(INPUT_INDEX["wx"]);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constWxAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsW = ge::OpDescUtils::MutableWeights(inputWNode);
  FUSION_PASS_CHECK(weightsW.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null, fusion failed."),
                    return FAILED);
  ge::GeTensorPtr wTensorPtr = weightsW[0];

  ge::InDataAnchorPtr inputRAnchorPtr0 = fused_node->GetInDataAnchor(INPUT_INDEX["wr"]);
  ge::OutDataAnchorPtr constRAnchorPtr0 = inputRAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputRNode = constRAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsR = ge::OpDescUtils::MutableWeights(inputRNode);
  FUSION_PASS_CHECK(weightsR.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsR is null, fusion failed."),
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

  int32_t hiddenSize = rRow / 4;

  // wxRow == whRow
  ge::GeTensorDesc weightTensorDesc;
  std::vector<int64_t> dimsIn = {wCol + rCol, wRow};
  SetTensorDescription(weightTensorDesc, dimsIn, ge::FORMAT_HWCN, dataType);

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
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.", fused_node->GetName().c_str());
    return FAILED;
  }
  
  return ret;
}

Status LSTMPFusionPass::AddEdgeForInput(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node) {

  auto fused_desc = fused_node->GetOpDesc();
  
  ge::NodePtr trans_x = nullptr;
  std::vector<int32_t> perm_x = {1, 0, 2};
  auto input_x_desc = fused_desc->GetInputDesc("x");
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_x_desc, trans_x, perm_x, "x") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "X CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(), trans_x->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x input edge to trans_x failed."),
        return FAILED);
        
  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_x->GetOutDataAnchor(0), dynamicv3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add trans_x edge to fusion node failed."),
        return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(INPUT_INDEX["bias"])->GetPeerOutAnchor(), dynamicv3_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add bias input edge to fusion node  failed."),
        return FAILED);

  ge::NodePtr trans_wp = nullptr;
  std::vector<int32_t> perm_p = {1, 0};
  auto input_desc = fused_desc->GetInputDesc("project");
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_desc, trans_wp, perm_p, "project") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(INPUT_INDEX["project"])->GetPeerOutAnchor(), trans_wp->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add project input edge to trans_wp failed."),
        return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_wp->GetOutDataAnchor(0), dynamicv3_node->GetInDataAnchor(3)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const project input edge to fusion node failed."),
        return FAILED);
  int src = 5;
  int dst = 4;
  if (fused_desc->MutableInputDesc("real_mask") == nullptr) {
    ge::NodePtr const_mask = nullptr;
    FUSION_PASS_CHECK(CreateConstNode(graph, fused_desc, const_mask) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateConstNode real_mask FAIL"),
                        return FAILED);
    FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(const_mask->GetOutDataAnchor(0), dynamicv3_node->GetInDataAnchor(dst)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const_mask input edge to fusion node failed."),
            return FAILED);
  } else {
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(INPUT_INDEX["real_mask"])->GetPeerOutAnchor(),
                                           dynamicv3_node->GetInDataAnchor(dst)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add real_mask input edge to fusion node  failed."),
        return FAILED);
  }
  
  std::vector<std::string> vec_input = {"init_h", "init_c"};

  for (auto& key : vec_input) {
    ++src;
    ++dst;
    if (fused_desc->MutableInputDesc(key) != nullptr) {
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(src)->GetPeerOutAnchor(),
                                       dynamicv3_node->GetInDataAnchor(dst)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add real_mask input edge to fusion node  failed."),
        return FAILED);
    }
  }
  return SUCCESS;
}

Status LSTMPFusionPass::AddEdgeForOutput(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node) {
  auto dynamicv3_desc = dynamicv3_node->GetOpDesc();
  ge::NodePtr trans_y = nullptr;
  std::vector<int32_t> perm = {1, 0, 2};
  auto input_y_desc = dynamicv3_desc->GetOutputDesc("y");
  FUSION_PASS_CHECK(CreateTransposeNode(graph, input_y_desc, trans_y, perm, "y") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Y CreateTransposeNode FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(0), trans_y->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x input edge to trans_x failed."),
        return FAILED);

  for (auto in_anchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for x fail"),
          return FAILED);
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_y->GetOutDataAnchor(0), in_anchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add trans_y edge to fusion node failed."),
        return FAILED);
  }

  ge::NodePtr splith_node = nullptr;

  FUSION_PASS_CHECK(CreateSplitNode(graph, dynamicv3_desc, splith_node, "output_h") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateSplitNode output_h FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_h"]), splith_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splith node  failed."),
        return FAILED);

  for (auto in_anchor : fused_node->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
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
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(OUTPUT_INDEX["output_c"]), splitc_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splitc node  failed."),
        return FAILED);

  for (auto in_anchor : fused_node->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(2), in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge for init_c fail"),
          return FAILED);
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(splitc_node->GetOutDataAnchor(1), in_anchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add splitc edge to fusion node failed."),
        return FAILED);
  }

  return SUCCESS;
}

Status LSTMPFusionPass::CreateSplitNode(ge::ComputeGraph& graph, ge::OpDescPtr& dynamicv3_desc, ge::NodePtr& new_node, const std::string& output_name) {
  auto input_desc = dynamicv3_desc->GetOutputDesc(output_name);
  std::shared_ptr<ge::OpDesc> split_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (split_desc = std::make_shared<ge::OpDesc>(output_name + "_split", "SplitVD")),
    return FAILED);
  
  FUSION_PASS_CHECK(split_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create split node"),
                    return FAILED);
  split_desc->AddInputDesc("x", input_desc);
  auto output_desc = input_desc.Clone();
  auto output_desc1 = input_desc.Clone();
  auto dims = output_desc.GetShape().GetDims();
  if (dims.size() != 3) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dim size should be 3,cur is %d",(int)dims.size());
    return FAILED;
  }
  std::vector<int64_t> new_dims = {dims[0] - 1, dims[1], dims[2]};
  output_desc.SetShape(ge::GeShape(new_dims));
  output_desc.SetOriginShape(ge::GeShape(new_dims));
  split_desc->AddOutputDesc("y0", output_desc);

  std::vector<int64_t> new_dims1 = {1, dims[1], dims[2]};
  output_desc1.SetShape(ge::GeShape(new_dims1));
  output_desc1.SetOriginShape(ge::GeShape(new_dims1));
  split_desc->AddOutputDesc("y1", output_desc1);

  ge::Operator::OpListInt size_split = {dims[0] - 1, 1};
  ge::AttrUtils::SetListInt(split_desc, "size_splits", size_split);
  ge::AttrUtils::SetInt(split_desc, "split_dim", 0);
  ge::AttrUtils::SetInt(split_desc, "num_split", 2);
  new_node = graph.AddNode(split_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateSplitNode node"),
                    return FAILED);
  return SUCCESS;
}

Status LSTMPFusionPass::RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node) {
  // unlink all control input of LSTMP
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
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", fused_node->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}
REGISTER_PASS("LSTMPFusionPass", BUILT_IN_GRAPH_PASS, LSTMPFusionPass);
} // fe