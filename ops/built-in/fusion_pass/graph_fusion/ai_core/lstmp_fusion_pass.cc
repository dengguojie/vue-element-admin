/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
                    Concat    DATA_PROJECT(可选)
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

  ge::NodePtr splith_node = nullptr;
  auto dynamicv3_desc = dynamicv3_node->GetOpDesc();
  FUSION_PASS_CHECK(CreateSplitNode(graph, dynamicv3_desc, splith_node, "output_h") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateSplitNode output_h FAIL"),
                    return FAILED);
  
  ge::NodePtr splitc_node = nullptr;
  FUSION_PASS_CHECK(CreateSplitNode(graph, dynamicv3_desc, splitc_node, "output_c") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateSplitNode output_c FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(), dynamicv3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fused 0 input edge to fusion node  failed."),
        return FAILED);
  
  auto index = CalcInputWxIndex(fused_desc);
  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(1), splith_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splith node  failed."),
        return FAILED);
  
  FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicv3_node->GetOutDataAnchor(2), splitc_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicv3 input edge to splitc node  failed."),
        return FAILED);

  FUSION_PASS_CHECK(AddEdgeForOptionInput(graph, fused_node, dynamicv3_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdgeForOptionInput FAIL"),
                    return FAILED);

  FUSION_PASS_CHECK(AddEdgeForOutput(fused_node, dynamicv3_node, splith_node, splitc_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdgeForOutput FAIL"),
                    return FAILED);
  
  FUSION_PASS_CHECK(RemoveFusedNode(graph, fused_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RemoveFusedNode FAIL"),
                    return FAILED);
  return SUCCESS;
}

int LSTMPFusionPass::CalcInputWxIndex(ge::OpDescPtr& fused_desc) {
  int index = 1;
  std::vector<std::string> input_name = {"real_mask", "init_h", "init_c"};
  for (auto& val : input_name) {
    if (fused_desc->MutableInputDesc(val) != nullptr) {
      ++index;
    }
  }
  return index;
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
  
  auto x_desc = fused_desc->GetInputDesc("x");
  dynamic_desc->AddInputDesc("x", x_desc);
  // w old
  // auto w_desc = trans_desc->GetOutputDesc(0);
  // w_desc.SetFormat(ge::FORMAT_HWCN);
  // w_desc.SetOriginFormat(ge::FORMAT_HWCN);
  // dynamic_desc->AddInputDesc("w", w_desc);
  // w new
  std::vector<ge::GeTensorPtr> wTensorPtrs;
  Status retW = ProcessLSTMWxh(fused_node, wTensorPtrs);
  FUSION_PASS_CHECK(retW != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
  dynamic_desc->AddInputDesc("w", wTensorPtrs[0]->GetTensorDesc());

  auto input_b_desc = fused_desc->GetInputDesc("x").Clone();
  std::vector<int64_t> dims = {wTensorPtrs[0]->GetTensorDesc().GetShape().GetDims().at(1)};
  input_b_desc.SetShape(ge::GeShape(dims));
  input_b_desc.SetOriginShape(ge::GeShape(dims));
  dynamic_desc->AddInputDesc("bias", input_b_desc);

  std::vector<std::string> vec_input = {"real_mask", "init_h", "init_c"};
  for (auto& key : vec_input) {
    if (fused_desc->MutableInputDesc(key) != nullptr) {
      auto input_desc = fused_desc->GetInputDesc(key);
      dynamic_desc->AddInputDesc(key, input_desc);
    }
  }

  if (fused_desc->MutableInputDesc("project") != nullptr) {
    auto input_wp = fused_desc->GetInputDesc("project");
    auto dims_wp = input_wp.GetShape().GetDims();
    FUSION_PASS_CHECK(dims_wp.size() != 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input wp dim_num should be 2, cur is[%d]", (int)dims_wp.size()),
                    return FAILED);
    std::vector<int64_t> dims_project = {dims_wp[1], dims_wp[0]};
    input_wp.SetShape(ge::GeShape(dims_project));
    input_wp.SetOriginShape(ge::GeShape(dims_project));
    dynamic_desc->AddInputDesc("project", input_wp);
  }
  auto outputx_desc = fused_desc->GetInputDesc("x").Clone();
  auto outputh_desc = outputx_desc.Clone();
  auto outputc_desc = outputx_desc.Clone();

  auto dims_x = outputx_desc.GetShape().GetDims();
  int seq = dims_x[0];
  int batch = dims_x[1];
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
  OP_LOGD("LSTMPFusionPass", "AddOutputDesc SUCCESS....");
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
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fused_node->GetInDataAnchor(4);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constWxAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsW = ge::OpDescUtils::MutableWeights(inputWNode);
  FUSION_PASS_CHECK(weightsW.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null, fusion failed."),
                    return FAILED);
  ge::GeTensorPtr wTensorPtr = weightsW[0];

  ge::InDataAnchorPtr inputRAnchorPtr0 = fused_node->GetInDataAnchor(5);
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


Status LSTMPFusionPass::CreateConstNode(ge::ComputeGraph&graph, ge::OpDescPtr& fused_desc, ge::NodePtr& new_node) {
  std::shared_ptr<ge::OpDesc> const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (const_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_const", "Const")),
    return FAILED);
  
  FUSION_PASS_CHECK(const_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);
  
  int len = fused_desc->GetInputDesc("wx").GetShape().GetDims().at(0);
  std::vector<int64_t>dims = {len};
  std::vector<float> val(len, 0);
  ge::GeShape shape(dims);
  ge::GeTensorDesc desc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::GeTensorPtr tensor_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                               desc, reinterpret_cast<uint8_t*>(val.data()), len * sizeof(float))),
                          tensor_ptr = nullptr;
                          return PARAM_INVALID);
  ge::AttrUtils::SetTensor(const_desc, "value", tensor_ptr);
  const_desc->AddOutputDesc(tensor_ptr->GetTensorDesc());
  new_node = graph.AddNode(const_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateConstNode node"),
                    return FAILED);
  return SUCCESS;
}

Status LSTMPFusionPass::AddEdgeForOptionInput(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node) {
  auto fused_desc = fused_node->GetOpDesc();
  auto b_index = CalcInputWxIndex(fused_desc) + 2;
  if (fused_desc->MutableInputDesc("bias") == nullptr) {
    ge::NodePtr const_b = nullptr;
    FUSION_PASS_CHECK(CreateConstNode(graph, fused_desc, const_b) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateConstNode b FAIL"),
                    return FAILED);
    
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(const_b->GetOutDataAnchor(0), dynamicv3_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const b  input edge to fusion node  failed."),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(b_index)->GetPeerOutAnchor(), dynamicv3_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fused 6 input edge to fusion node  failed."),
        return FAILED);
  }
  
  int src_index = 0;
  int dst_index = 2;
  std::vector<std::string> input_name = {"real_mask", "init_h", "init_c"};
  for (auto& val : input_name) {
    if (fused_desc->MutableInputDesc(val) != nullptr) {
      ++src_index;
      ++dst_index;
      FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(src_index)->GetPeerOutAnchor(),
                                           dynamicv3_node->GetInDataAnchor(dst_index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fused  input edge to fusion node  failed."),
        return FAILED);
    }
  }

  src_index = b_index;
  if (fused_desc->MutableInputDesc("project") != nullptr) {
    ++src_index;
    ++dst_index;
    ge::NodePtr trans_wp = nullptr;
    std::vector<int32_t> perm = {1, 0};
    auto input_desc = fused_desc->GetInputDesc("project");
    FUSION_PASS_CHECK(CreateTransposeNode(graph, input_desc, trans_wp, perm, "project") != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateTransposeNode FAIL"),
                    return FAILED);
    
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(trans_wp->GetOutDataAnchor(0), dynamicv3_node->GetInDataAnchor(dst_index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const b  input edge to fusion node  failed."),
        return FAILED);
    
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(src_index)->GetPeerOutAnchor(), trans_wp->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const b  input edge to fusion node  failed."),
        return FAILED);
  }
  return SUCCESS;
}

Status LSTMPFusionPass::AddEdgeForOutput(ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node, ge::NodePtr& splith_node, ge::NodePtr& splitc_node) {
  std::vector<ge::NodePtr> list_node = {dynamicv3_node, splith_node, splitc_node};
  int out_index = 0;
  for (int i = 0; i < 3; ++i) {
    if (i != 0) {
      out_index = 1;
    }
    auto output_anchor0 = fused_node->GetOutDataAnchor(i);
    auto dynamic_outanchor = list_node[i]->GetOutDataAnchor(out_index);
    for (auto in_anchor : output_anchor0->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(output_anchor0, in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge outanchor 0 fail %d", i),
          return FAILED);
      
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamic_outanchor, in_anchor),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from outanchor 0 fail %d", i),
          return FAILED);
    }
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
  // unlink all control input of DynamicRNN
  if (fused_node->GetInControlAnchor() != nullptr) {
    fused_node->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of DynamicRNN
  for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  // unlink all output
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