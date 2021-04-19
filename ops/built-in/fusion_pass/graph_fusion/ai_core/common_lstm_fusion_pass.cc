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

#include "common_lstm_fusion_pass.h"
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
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "CommonLSTM";
static const std::string PATTERN_FUSEDNODE = "CommonLSTM";

vector<FusionPattern *> CommonLSTMFusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("CommonLSTMFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                "common lstm pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  return patterns;
}

template <class T>
static ge::GeTensorPtr SetWeightTensorData(ge::GeTensorPtr wTensorPtr, ge::GeTensorPtr rTensorPtr,
                                           std::vector<int32_t> &inputDims, ge::GeTensorDesc tensorDesc) {
  int32_t wRow = inputDims[0];
  int32_t wCol = inputDims[1];
  int32_t rRow = inputDims[2];
  int32_t rCol = inputDims[3];
  int32_t targetCol = wCol + rCol;
  int32_t weightSize = targetCol * wRow;
  // the wx + wh matrix
  unique_ptr<T[]> wxhMergeData(new (std::nothrow) T[weightSize]());
  FUSION_PASS_CHECK(wxhMergeData.get() == nullptr, OP_LOGE(FUSED_NODE, "wxhMergeData is NULL"), return nullptr);
  T *wxData = (T *)wTensorPtr->GetData().data();
  T *whData = (T *)rTensorPtr->GetData().data();

  auto retMem = memset_s(wxhMergeData.get(), weightSize, 0, weightSize);
  FUSION_PASS_CHECK(retMem != EOK, OP_LOGE(FUSED_NODE, "Failed to operate memset_s function!"), return nullptr);

  // wx transpose, assign to merge data
  T *dstWeight = wxhMergeData.get();
  for (int32_t i = 0; i < wRow * wCol; ++i) {
    *(dstWeight + i / wCol + wRow * (i % wCol)) = *(wxData + i);
  }

  // wh transpose, assign to merge data
  for (int32_t i = 0; i < rRow * rCol; ++i) {
    *(dstWeight + wRow * wCol + i / rCol + rRow * (i % rCol)) = *(whData + i);
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
                          return nullptr);
  return dstTensorPtr;
}

ge::GeTensorPtr CommonLSTMFusionPass::ProcessLSTMWxh(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                                     int32_t &hiddenSize) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "has enter process onnx lstm W");
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.wIndex);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constWxAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsW = ge::OpDescUtils::MutableWeights(inputWNode);
  FUSION_PASS_CHECK(weightsW.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null, fusion failed."),
                    return nullptr);
  ge::GeTensorPtr wTensorPtr = weightsW[0];

  ge::InDataAnchorPtr inputRAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.rIndex);
  ge::OutDataAnchorPtr constRAnchorPtr0 = inputRAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputRNode = constRAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsR = ge::OpDescUtils::MutableWeights(inputRNode);
  FUSION_PASS_CHECK(weightsR.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "LSTM weightsR is null, fusion failed."),
                    return nullptr);
  ge::GeTensorPtr rTensorPtr = weightsR[0];

  ge::GeTensorDesc wConstTensorDesc = wTensorPtr->GetTensorDesc();
  ge::GeTensorDesc rConstTensorDesc = rTensorPtr->GetTensorDesc();

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  int32_t wRow = wConstTensorDesc.GetShape().GetDim(1);
  int32_t wCol = wConstTensorDesc.GetShape().GetDim(2);
  int32_t rRow = rConstTensorDesc.GetShape().GetDim(1);
  int32_t rCol = rConstTensorDesc.GetShape().GetDim(2);
  FUSION_PASS_CHECK(wCol == 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "wCol can not 0"), return nullptr);
  FUSION_PASS_CHECK(rCol == 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "rCol can not 0"), return nullptr);

  hiddenSize = rRow / 4;

  // wxRow == whRow
  ge::GeTensorDesc weightTensorDesc;
  std::vector<int64_t> dimsIn = {wCol + rCol, wRow};
  SetTensorDescription(weightTensorDesc, dimsIn, ge::FORMAT_HWCN, dataType);

  ge::GeTensorPtr dstTensorPtr = nullptr;
  std::vector<int32_t> inputDims{wRow, wCol, rRow, rCol};
  if (dataType == ge::DT_FLOAT16) {
    dstTensorPtr = SetWeightTensorData<uint16_t>(wTensorPtr, rTensorPtr, inputDims, weightTensorDesc);
  } else if (dataType == ge::DT_FLOAT) {
    dstTensorPtr = SetWeightTensorData<float>(wTensorPtr, rTensorPtr, inputDims, weightTensorDesc);
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.");
    return nullptr;
  }
  return dstTensorPtr;
}

template <class T>
static ge::GeTensorPtr SetBiasTensorData(ge::GeTensorPtr srcBiasTensorPtr, ge::GeTensorDesc tensorDesc,
                                         int32_t hiddenSize, DataType dataType) {
  int32_t biasSize = 4 * hiddenSize;
  unique_ptr<T[]> dstBiasData(new (std::nothrow) T[biasSize]());
  auto retMem = memset_s(dstBiasData.get(), biasSize, 0, biasSize);
  FUSION_PASS_CHECK(retMem != EOK, OP_LOGE(FUSED_NODE, "bias memset failed!"), return nullptr);

  if (srcBiasTensorPtr != nullptr) {
    T *biasData = (T *)srcBiasTensorPtr->GetData().data();
    T *dstBias = dstBiasData.get();
    if (dataType == ge::DT_FLOAT16) {
      fp16_t totalValue;
      for (int32_t i = 0; i < biasSize; i++) {
        fp16_t value1(*(biasData + i));
        fp16_t value2(*(biasData + biasSize + i));
        totalValue = value1 + value2;
        dstBias[i] = totalValue.val;
      }
    } else {
      for (int32_t i = 0; i < biasSize; i++) {
        dstBias[i] = *(biasData + i) + *(biasData + biasSize + i);
      }
    }

    // swap 1, 3
    for (int32_t i = 0; i < hiddenSize; i++) {
      T tmp = dstBias[i + hiddenSize * 1];
      dstBias[i + hiddenSize * 1] = dstBias[i + hiddenSize * 3];
      dstBias[i + hiddenSize * 3] = tmp;
    }
  }

  ge::GeTensorPtr dstTensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((dstTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(dstBiasData.get()), biasSize * sizeof(T))),
                          return nullptr);
  return dstTensorPtr;
}

ge::GeTensorPtr CommonLSTMFusionPass::ProcessLSTMBias(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                                      int32_t hiddenSize) {
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  ge::GeTensorDesc biasTensorDesc;
  std::vector<int64_t> dimsIn = {4 * hiddenSize};
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  SetTensorDescription(biasTensorDesc, dimsIn, ge::FORMAT_ND, dataType);

  bool hasBias = fusedDesc->MutableInputDesc("b") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "has enter process onnx LSTM bias has bool %d", hasBias);
  ge::GeTensorPtr srcTensorPtr = nullptr;
  if (hasBias) {
    ge::InDataAnchorPtr biasInputAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.biasIndex);
    ge::OutDataAnchorPtr constBiasAnchorPtr0 = biasInputAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr biasNode = constBiasAnchorPtr0->GetOwnerNode();
    vector<ge::GeTensorPtr> biasT = ge::OpDescUtils::MutableWeights(biasNode);
    FUSION_PASS_CHECK(biasT.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "onnx LSTM biasT is null, fusion failed."),
                      return nullptr);

    srcTensorPtr = biasT[0];
    FUSION_PASS_CHECK(srcTensorPtr->GetTensorDesc().GetShape().GetDim(0) != 1,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Currently, the bias size cannot be set to 2. fusion failed."),
                      return nullptr);
  }

  ge::GeTensorPtr dstTensorPtr = nullptr;
  if (dataType == ge::DT_FLOAT16) {
    dstTensorPtr = SetBiasTensorData<uint16_t>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType);
  } else if (dataType == ge::DT_FLOAT) {
    dstTensorPtr = SetBiasTensorData<float>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType);
  }
  return dstTensorPtr;
}

void CommonLSTMFusionPass::SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims,
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

Status CommonLSTMFusionPass::AddReshapeNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode,
                                            ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes,
                                            std::string nodeName, int nodeIndex) {
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string operatorName = fusedDesc->GetName() + "/Reshape" + nodeName;
  auto reshapeOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "Reshape");
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "create Reshape Op operator error"),
                    return FAILED);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(0);
  reshape_desc->UpdateInputDesc("x", dynamicRnnOutputDesc);
  reshape_desc->UpdateInputDesc("shape", originTensorDesc);
  reshape_desc->UpdateOutputDesc("y", originTensorDesc);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(myReshape_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "myReshape_node node is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnNode->GetOutDataAnchor(nodeIndex),
                                                        myReshape_node->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion myReshape_node x failed."),
                    return FAILED);

  newNodes.push_back(myReshape_node);
  // Get Output Node
  for (InDataAnchorPtr oriTopPeerAnchorPtri : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtri->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(myReshape_node->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add Reshape Node edge to fusion node output %s failed.", nodeName.c_str()),
        return FAILED);
  }
  return SUCCESS;
}

Status CommonLSTMFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of LSTM
  OP_LOGI(FUSED_OP_TYPE.c_str(), "common lstm start fusion");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode is null, fusion failed."),
  return PARAM_INVALID);

  // get the OpDescPtr of LSTM
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode OpDesc is null, fusion failed."),
  return PARAM_INVALID);

  auto dynamicRnnOp = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/DynamicRnn", "DynamicRNN");
  FUSION_PASS_CHECK(dynamicRnnOp.IsEmpty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "create DynamicRnn operator error"),
  return FAILED);
  auto dynamicRnnDesc = ge::OpDescUtils::GetOpDescFromOperator(dynamicRnnOp);
  dynamicRnnOp.BreakConnect();

  // process x
  ge::GeTensorDesc xDesc = fusedDesc->GetInputDesc(0).Clone();
  dynamicRnnDesc->UpdateInputDesc("x", xDesc);

  // process seq_length
  bool hasSeqLength = fusedDesc->MutableInputDesc("sequence_lens") != nullptr;
  if (hasSeqLength) {
    ge::GeTensorDesc seq_length_desc = *fusedDesc->MutableInputDesc("sequence_lens");
    dynamicRnnDesc->UpdateInputDesc("seq_length", seq_length_desc);
  }

  // process init_h
  bool hasInitH = fusedDesc->MutableInputDesc("initial_h") != nullptr;
  if (hasInitH) {
    ge::GeTensorDesc initial_h_desc = *fusedDesc->MutableInputDesc("initial_h");
    dynamicRnnDesc->UpdateInputDesc("init_h", initial_h_desc);
  }

  // process init_c
  bool hasInitC = fusedDesc->MutableInputDesc("initial_c") != nullptr;
  if (hasInitC) {
    ge::GeTensorDesc initial_c_desc = *fusedDesc->MutableInputDesc("initial_c");
    dynamicRnnDesc->UpdateInputDesc("init_c", initial_c_desc);
  }

  // process w
  InputIndexInfo inputIndexInfo;
  int32_t hiddenSize = 0;
  ge::GeTensorPtr wTensorPtr = ProcessLSTMWxh(fusedNode, inputIndexInfo, hiddenSize);
  FUSION_PASS_CHECK(wTensorPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
  dynamicRnnDesc->UpdateInputDesc("w", wTensorPtr->GetTensorDesc());

  // process bias
  ge::GeTensorPtr biasTensorPtr = ProcessLSTMBias(fusedNode, inputIndexInfo, hiddenSize);
  FUSION_PASS_CHECK(biasTensorPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Process b fail."), return FAILED);
  dynamicRnnDesc->AddInputDesc("b", biasTensorPtr->GetTensorDesc());

  // create dynamic_rnn node
  ge::NodePtr dynamicRnnNode = graph.AddNode(dynamicRnnDesc);
  FUSION_PASS_CHECK(dynamicRnnNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "dynamicLSTM node is null, fusion failed."),
  return FAILED);
  newNodes.push_back(dynamicRnnNode);

  // connect x
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       dynamicRnnNode->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node x failed."),
  return FAILED);

  // connect w
  ge::OpDescPtr wDesc = ge::OpDescUtils::CreateConstOp(wTensorPtr);
  ge::NodePtr wNode = graph.AddNode(wDesc);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(wNode->GetOutDataAnchor(0), dynamicRnnNode->GetInDataAnchor(1)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node w failed."), return FAILED);

  // connect bias
  ge::OpDescPtr biasDesc = ge::OpDescUtils::CreateConstOp(biasTensorPtr);
  ge::NodePtr biasNode = graph.AddNode(biasDesc);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(biasNode->GetOutDataAnchor(0), dynamicRnnNode->GetInDataAnchor(2)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node bias failed."), return FAILED);

  // connect seq_length
  if (hasSeqLength) {
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                            dynamicRnnNode->GetInDataAnchor(3));
  }

  // connect init_h
  if (hasInitH) {
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                            dynamicRnnNode->GetInDataAnchor(4));
  }

  // connect init_c
  if (hasInitC) {
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                            dynamicRnnNode->GetInDataAnchor(5));
  }

  // use common rnn output y
  ge::GeTensorDesc outputYDesc = fusedDesc->GetOutputDesc(0).Clone();
  std::vector<int64_t> dims = outputYDesc.GetShape().GetDims();

  // t derec batch hiden
  dims.erase(std::begin(dims) + 1);
  ge::GeShape y_shape(dims);
  outputYDesc.SetShape(y_shape);

  // use common rnn y update all output default value
  dynamicRnnDesc->UpdateOutputDesc("y", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("output_h", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("output_c", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("i", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("j", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("f", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("o", outputYDesc);
  dynamicRnnDesc->UpdateOutputDesc("tanhc", outputYDesc);

  FUSION_PASS_CHECK(AddReshapeNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "Y", 0) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ReshapeY Op operator error"), return FAILED);

  FUSION_PASS_CHECK(AddReshapeNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "H", 1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ReshapeH Op operator error"), return FAILED);

  FUSION_PASS_CHECK(AddReshapeNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "C", 2) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "create ReshapeC Op operator error"), return FAILED);

  // unlink all control input of DynamicRNN
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of DynamicRNN
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // unlink all output
  for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  // remove LSTM from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
  return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "common lstm end fusion");
  return SUCCESS;
}

REGISTER_PASS("CommonLSTMFusionPass", BUILT_IN_GRAPH_PASS, CommonLSTMFusionPass);
} // namespace fe
