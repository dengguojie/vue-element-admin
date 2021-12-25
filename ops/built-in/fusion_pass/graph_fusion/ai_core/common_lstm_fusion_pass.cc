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
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "CommonLSTM";
static const std::string PATTERN_FUSEDNODE = "CommonLSTM";
static const string DIRECTION = "direction";
static const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";

vector<FusionPattern *> CommonLSTMFusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("CommonLSTMFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                "common lstm pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  return patterns;
}

template <class T>
static Status SetInitHCTensorData(ge::GeTensorPtr initTensorPtr, ge::GeTensorDesc tensorDesc,
                                  int32_t &init_size, vector<ge::GeTensorPtr> &tensorPtr) {
  unique_ptr<T[]> initData(new (std::nothrow) T[init_size]());
  FUSION_PASS_CHECK(initData.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "initdata is null"),
  return FAILED);
  unique_ptr<T[]> initDataReverse(new (std::nothrow) T[init_size]());
  FUSION_PASS_CHECK(initDataReverse.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "initDataReverse is null"),
  return FAILED);
  T *srcData = (T *)initTensorPtr->GetData().data();
  FUSION_PASS_CHECK(srcData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "initTensorPtr->GetData().data() is NULL"),
                    return FAILED);

  auto retMem = memset_s(initData.get(), init_size, 0, init_size);
  FUSION_PASS_CHECK(retMem != EOK,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "failed to operate memset_s function!"),
  return FAILED);
  retMem = memset_s(initDataReverse.get(), init_size, 0, init_size);
  FUSION_PASS_CHECK(retMem != EOK,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "failed to operate memset_s function!"),
  return FAILED);

  FUSION_PASS_CHECK(initData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "initData is NULL"),
                    return FAILED);
  FUSION_PASS_CHECK(initDataReverse.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "initDataReverse is NULL"),
                    return FAILED);

  T *dstData = initData.get();
  T *dstDataR = initDataReverse.get();

  for (int32_t i = 0; i < init_size; i++) {
    *(dstData + i) = *(srcData + i);
    *(dstDataR + i) = *(srcData + i + init_size);
  }

  ge::GeTensorPtr initPtrForward = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (initPtrForward =
          std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(initData.get()),
                                           init_size * sizeof(T))),
      return FAILED);
  initPtrForward->SetTensorDesc(tensorDesc);
  tensorPtr.push_back(initPtrForward);

  ge::GeTensorPtr initPtrReverse = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (initPtrReverse =
          std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(initDataReverse.get()),
                                           init_size * sizeof(T))),
      return FAILED);
  initPtrReverse->SetTensorDesc(tensorDesc);
  tensorPtr.push_back(initPtrReverse);
  return SUCCESS;
}

Status CommonLSTMFusionPass::ProcessLSTMInitH(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                              vector<ge::GeTensorPtr> &tensorPtr) {
  ge::OutDataAnchorPtr initHAnchorPtr = fusedNode->GetInDataAnchor(inputIndexInfo.inithIndex)->GetPeerOutAnchor();
  ge::NodePtr initHNode = initHAnchorPtr->GetOwnerNode();
  vector<ge::GeTensorPtr> initH = ge::OpDescUtils::MutableWeights(initHNode);
  if (initH.empty()) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM initH is null ,fusion failed");
    return FAILED;
  }

  ge::GeTensorPtr initHPtr = initH[0];
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  ge::GeTensorDesc initHDesc = fusedDesc->GetInputDesc(inputIndexInfo.inithIndex).Clone();
  vector<int64_t> dims = initHDesc.GetShape().GetDims();
  int32_t init_size = dims[1] * dims[2];

  dims[0] = 1;
  ge::GeShape init_shape(dims);
  initHDesc.SetShape(init_shape);
  initHDesc.SetOriginShape(init_shape);

  Status ret = SUCCESS;
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  if (dataType == ge::DT_FLOAT16) {
    ret = SetInitHCTensorData<uint16_t>(initHPtr, initHDesc, init_size, tensorPtr);
  } else if (dataType == ge::DT_FLOAT) {
    ret = SetInitHCTensorData<float>(initHPtr, initHDesc, init_size, tensorPtr);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.", fusedNode->GetName().c_str());
    return FAILED;
  }
  return ret;
}

Status CommonLSTMFusionPass::ProcessLSTMInitC(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                              vector<ge::GeTensorPtr> &tensorPtr) {
  ge::OutDataAnchorPtr initCAnchorPtr = fusedNode->GetInDataAnchor(inputIndexInfo.initcIndex)->GetPeerOutAnchor();
  ge::NodePtr initCNode = initCAnchorPtr->GetOwnerNode();
  vector<ge::GeTensorPtr> initC = ge::OpDescUtils::MutableWeights(initCNode);
  if (initC.empty()) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM initC is null ,fusion failed");
    return FAILED;
  }

  ge::GeTensorPtr initCPtr = initC[0];
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  ge::GeTensorDesc initCDesc = fusedDesc->GetInputDesc(inputIndexInfo.initcIndex).Clone();
  vector<int64_t> dims = initCDesc.GetShape().GetDims();
  int32_t init_size = dims[1] * dims[2];

  dims[0] = 1;
  ge::GeShape init_shape(dims);
  initCDesc.SetShape(init_shape);
  initCDesc.SetOriginShape(init_shape);

  Status ret;
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  if (dataType == ge::DT_FLOAT16) {
    ret = SetInitHCTensorData<uint16_t>(initCPtr, initCDesc, init_size, tensorPtr);
  } else if (dataType == ge::DT_FLOAT) {
    ret = SetInitHCTensorData<float>(initCPtr, initCDesc, init_size, tensorPtr);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.", fusedNode->GetName().c_str());
    return FAILED;
  }
  return ret;
}

template <class T>
static Status SetBiasTensorData(ge::GeTensorPtr srcBiasTensorPtr, ge::GeTensorDesc tensorDesc, int32_t hiddenSize,
                                DataType dataType, int32_t &start_size, vector<ge::GeTensorPtr> &tensorPtr) {
  int32_t biasSize = 4 * hiddenSize;
  unique_ptr<T[]> dstBiasData(new (std::nothrow) T[biasSize]());
  auto retMem = memset_s(dstBiasData.get(), biasSize, 0, biasSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "bias memset failed!"), return FAILED);

  if (srcBiasTensorPtr == nullptr) {
    ge::GeTensorPtr dstTensorPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (dstTensorPtr = std::make_shared<ge::GeTensor>(
              tensorDesc, reinterpret_cast<uint8_t *>(dstBiasData.get()), biasSize * sizeof(T))),
        return FAILED);
    tensorPtr.push_back(dstTensorPtr);
    return SUCCESS;
  }
  T *biasData = (T *)srcBiasTensorPtr->GetData().data();
  T *dstBias = dstBiasData.get();
  if (dataType == ge::DT_FLOAT16) {
    fp16_t totalValue;
    for (int32_t i = 0; i < biasSize; i++) {
      fp16_t value1(*(biasData + i + start_size));
      fp16_t value2(*(biasData + biasSize + i + start_size));
      totalValue = value1 + value2;
      dstBias[i] = totalValue.val;
    }
  } else {
    for (int32_t i = 0; i < biasSize; i++) {
      dstBias[i] = *(biasData + i + start_size) + *(biasData + biasSize + i + start_size);
    }
  }

  // swap 1, 3
  for (int32_t i = 0; i < hiddenSize; i++) {
    T tmp = dstBias[i + hiddenSize * 1];
    dstBias[i + hiddenSize * 1] = dstBias[i + hiddenSize * 3];
    dstBias[i + hiddenSize * 3] = tmp;
  }
  ge::GeTensorPtr dstTensorPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((dstTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(dstBiasData.get()), biasSize * sizeof(T))),
                          return FAILED);
  tensorPtr.push_back(dstTensorPtr);
  return SUCCESS;
}

Status CommonLSTMFusionPass::ProcessLSTMBias(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                             int32_t num_directions, int32_t hiddenSize,
                                             vector<ge::GeTensorPtr> &tensorPtr) {
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
    FUSION_PASS_CHECK(biasT.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "onnx LSTM biasT is null, fusion failed."),
                      return FAILED);
    srcTensorPtr = biasT[0];
  }
  Status ret = SUCCESS;
  if (num_directions == 1) {
    int32_t starts_size = 0;
    if (dataType == ge::DT_FLOAT16) {
      ret = SetBiasTensorData<uint16_t>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size, tensorPtr);
    } else if (dataType == ge::DT_FLOAT) {
      ret = SetBiasTensorData<float>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size, tensorPtr);
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s 's type is not in (float16, float32), fusion failed", fusedNode->GetName().c_str());
    }
  } else if (num_directions == 2) {
    int32_t starts_size1 = 0;
    int32_t starts_size2 = 8 * hiddenSize;
    if (dataType == ge::DT_FLOAT16) {
      ret = SetBiasTensorData<uint16_t>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size1, tensorPtr);
      ret = SetBiasTensorData<uint16_t>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size2, tensorPtr);
    } else if (dataType == ge::DT_FLOAT) {
      ret = SetBiasTensorData<float>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size1, tensorPtr);
      ret = SetBiasTensorData<float>(srcTensorPtr, biasTensorDesc, hiddenSize, dataType, starts_size2, tensorPtr);
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s 's type is not in (float16, float32), fusion failed", fusedNode->GetName().c_str());
    }
  }
  return ret;
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
  FUSION_PASS_CHECK(wxhMergeData.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxhMergeData is NULL"),
                    return FAILED);
  T *wxData = (T *)wTensorPtr->GetData().data();
  T *whData = (T *)rTensorPtr->GetData().data();
  FUSION_PASS_CHECK(wxData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "wxData is NULL"), return FAILED);
  FUSION_PASS_CHECK(whData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "whData is NULL"), return FAILED);

  auto retMem = memset_s(wxhMergeData.get(), weightSize, 0, weightSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_NODE, "Failed to operate memset_s function!"),
                    return FAILED);

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

Status CommonLSTMFusionPass::ProcessLSTMWxh(ge::NodePtr fusedNode, const InputIndexInfo &inputIndexInfo,
                                            int32_t &hiddenSize, int32_t num_directions,
                                            vector<ge::GeTensorPtr> &tensorPtr) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "has enter process onnx lstm W");
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.wIndex);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constWxAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsW = ge::OpDescUtils::MutableWeights(inputWNode);
  FUSION_PASS_CHECK(weightsW.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsW is null, fusion failed."),
                    return FAILED);
  ge::GeTensorPtr wTensorPtr = weightsW[0];

  ge::InDataAnchorPtr inputRAnchorPtr0 = fusedNode->GetInDataAnchor(inputIndexInfo.rIndex);
  ge::OutDataAnchorPtr constRAnchorPtr0 = inputRAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputRNode = constRAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsR = ge::OpDescUtils::MutableWeights(inputRNode);
  FUSION_PASS_CHECK(weightsR.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsR is null, fusion failed."),
                    return FAILED);
  ge::GeTensorPtr rTensorPtr = weightsR[0];

  ge::GeTensorDesc wConstTensorDesc = wTensorPtr->GetTensorDesc();
  ge::GeTensorDesc rConstTensorDesc = rTensorPtr->GetTensorDesc();

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(0).GetDataType();
  int32_t wRow = wConstTensorDesc.GetShape().GetDim(1);
  int32_t wCol = wConstTensorDesc.GetShape().GetDim(2);
  int32_t rRow = rConstTensorDesc.GetShape().GetDim(1);
  int32_t rCol = rConstTensorDesc.GetShape().GetDim(2);
  FUSION_PASS_CHECK(wCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "wCol can not 0"), return FAILED);
  FUSION_PASS_CHECK(rCol == 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "rCol can not 0"), return FAILED);

  hiddenSize = rRow / 4;

  // wxRow == whRow
  ge::GeTensorDesc weightTensorDesc;
  std::vector<int64_t> dimsIn = {wCol + rCol, wRow};
  SetTensorDescription(weightTensorDesc, dimsIn, ge::FORMAT_HWCN, dataType);

  std::vector<int32_t> inputDims{wRow, wCol, rRow, rCol};
  Status ret = SUCCESS;
  if (num_directions == 1) {
    std::vector<int32_t> start_size = {0, 0};
    if (dataType == ge::DT_FLOAT16) {
      ret = SetWeightTensorData<uint16_t>(wTensorPtr, rTensorPtr, inputDims,
                                          weightTensorDesc, start_size, tensorPtr);
    } else if (dataType == ge::DT_FLOAT) {
      ret = SetWeightTensorData<float>(wTensorPtr, rTensorPtr, inputDims,
                                       weightTensorDesc, start_size, tensorPtr);
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.", fusedNode->GetName().c_str());
      return FAILED;
    }
  } else if (num_directions == 2) {
    std::vector<int32_t> start_size1 = {0, 0};
    std::vector<int32_t> start_size2 = {wRow * wCol, rRow * rCol};
    if (dataType == ge::DT_FLOAT16) {
      ret = SetWeightTensorData<uint16_t>(wTensorPtr, rTensorPtr, inputDims,
                                          weightTensorDesc, start_size1, tensorPtr);
      ret = SetWeightTensorData<uint16_t>(wTensorPtr, rTensorPtr, inputDims,
                                          weightTensorDesc, start_size2, tensorPtr);
    } else if (dataType == ge::DT_FLOAT) {
      ret = SetWeightTensorData<float>(wTensorPtr, rTensorPtr, inputDims,
                                       weightTensorDesc, start_size1, tensorPtr);
      ret = SetWeightTensorData<float>(wTensorPtr, rTensorPtr, inputDims,
                                       weightTensorDesc, start_size2, tensorPtr);
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.", fusedNode->GetName().c_str());
      return FAILED;
    }
  }
  return ret;
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
  FUSION_PASS_CHECK(reshapeOp.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create Reshape Op operator error"),
                    return FAILED);
  auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
  reshapeOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(0);
  reshape_desc->UpdateInputDesc("x", dynamicRnnOutputDesc);
  reshape_desc->UpdateInputDesc("shape", originTensorDesc);
  reshape_desc->UpdateOutputDesc("y", originTensorDesc);

  ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(myReshape_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "myReshape_node node is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnNode->GetOutDataAnchor(nodeIndex),
                                                        myReshape_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion myReshape_node x failed."),
                    return FAILED);

  newNodes.push_back(myReshape_node);
  // Get Output Node
  for (InDataAnchorPtr oriTopPeerAnchorPtri : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtri->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(myReshape_node->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add Reshape Node edge to fusion node output %s failed.", nodeName.c_str()),
        return FAILED);
  }
  return SUCCESS;
}

Status CommonLSTMFusionPass::AddExpandDimsNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode,
                                               ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes,
                                               std::string nodeName, int nodeIndex) {
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string operatorName = fusedDesc->GetName() + "/ExpandDims" + nodeName;
  auto ExpandDimOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "ExpandDims");
  FUSION_PASS_CHECK(ExpandDimOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDims Op operator error"),
                    return FAILED);
  auto ExpandDimDesc = ge::OpDescUtils::GetOpDescFromOperator(ExpandDimOp);
  ExpandDimOp.BreakConnect();

  ge::GeTensorDesc tensorDesc;
  vector<int64_t> dimsIn = {1};
  SetTensorDescription(tensorDesc, dimsIn, ge::FORMAT_ND, ge::DT_INT32);

  ge::GeTensorDesc ExdTensorDesc = fusedDesc->GetOutputDesc(0).Clone();
  ExpandDimDesc->UpdateInputDesc("x", dynamicRnnOutputDesc);
  ExpandDimDesc->UpdateInputDesc("axis", tensorDesc);
  ExpandDimDesc->UpdateOutputDesc("y", ExdTensorDesc);

  ge::GeTensorPtr AxisTensorPtr = nullptr;
  vector<int32_t> axis = {1};
  FUSION_PASS_MAKE_SHARED((AxisTensorPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t *>(axis.data()), 1 * sizeof(int32_t))),
                          return FAILED);
  ge::OpDescPtr AxisDesc = ge::OpDescUtils::CreateConstOp(AxisTensorPtr);

  ge::NodePtr ExpandDimNode = graph.AddNode(ExpandDimDesc);
  FUSION_PASS_CHECK(ExpandDimNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ExpandDim node is null, fusion failed."),
                    return FAILED);
  ge::NodePtr AxisNode = graph.AddNode(AxisDesc);
  FUSION_PASS_CHECK(AxisNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Axis node is null, fusion failed."),
                    return FAILED);

  vector<std::string> original_names = {"axis"};
  bool ret = ge::AttrUtils::SetListStr(ExpandDimNode->GetOpDesc(), ATTR_NAME_OP_INFER_DEPENDS, original_names);
  if (!ret) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ExpandDimNode set ATTR_NAME_OP_INFER_DEPENDS error.");
  }

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnNode->GetOutDataAnchor(nodeIndex),
                                                       ExpandDimNode->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion ExpandDim x failed."),
        return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(AxisNode->GetOutDataAnchor(0),
                                                       ExpandDimNode->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add axis edge to fusion ExpandDim axis failed."),
        return FAILED);

  newNodes.push_back(AxisNode);
  newNodes.push_back(ExpandDimNode);

  // get output node
  for (InDataAnchorPtr oriTopPeerAnchorPtri : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtri->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(ExpandDimNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "add ExpandDim Node edge to fusion node output %s failed.", nodeName.c_str()),
        return FAILED);
  }

  return SUCCESS;
}

Status CommonLSTMFusionPass::AddRNNMaskNode(ge::NodePtr fusedNode, ge::NodePtr dynamicRnnNode, ge::ComputeGraph &graph,
                                            int32_t hiddenSize, vector<ge::NodePtr> &newNodes)
    {
        int32_t seqLenIndex = 4;
        bool rnnGenMaskExist = false;
        ge::NodePtr existRnnNode = nullptr;
        auto outDataAnchor = fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor();
        for (auto nextInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
            ge::NodePtr outputNode = nextInDataAnchor->GetOwnerNode();
            if (outputNode->GetType() == "RnnGenMask") {
                rnnGenMaskExist = true;
                existRnnNode = outputNode;
                break;
            }
        }
        if (rnnGenMaskExist) {
            ge::GeTensorDesc tensorOutDesc = existRnnNode->GetOpDesc()->GetOutputDesc(0).Clone();
            dynamicRnnNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutDesc);
            // Add Edge
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(existRnnNode->GetOutDataAnchor(0),
                dynamicRnnNode->GetInDataAnchor(3)),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);
            return SUCCESS;
        }

        ge::OpDescPtr rnnMaskDesc = nullptr;
        FUSION_PASS_MAKE_SHARED(
            (rnnMaskDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/RnnGenMask", "RnnGenMask")),
            rnnMaskDesc = nullptr; return FAILED);
        ge::GeTensorDesc inputRnnMaskDesc = fusedNode->GetOpDesc()->GetInputDesc(seqLenIndex).Clone();
        std::vector <int64_t> dimLength = inputRnnMaskDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(dimLength.size() != 1,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Unexcepted seqlength input shape"), return FAILED);
        int64_t batchSize = dimLength[0];
        int64_t numStep = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0);
        std::vector <int64_t> maskDims = {numStep, batchSize, hiddenSize};
        ge::GeShape tensorMaskShape(maskDims);
        ge::GeShape tensorMaskOriginShape(maskDims);
        ge::GeTensorDesc tensorOutputMaskDesc = ge::GeTensorDesc(tensorMaskShape, ge::FORMAT_ND, ge::DT_FLOAT16);
        tensorOutputMaskDesc.SetOriginShape(tensorMaskOriginShape);
        tensorOutputMaskDesc.SetOriginFormat(ge::FORMAT_ND);
        rnnMaskDesc->AddInputDesc("seq_length", inputRnnMaskDesc);
        rnnMaskDesc->AddOutputDesc("seq_mask", tensorOutputMaskDesc);
        dynamicRnnNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutputMaskDesc);

        // Set Attr
        ge::AttrUtils::SetInt(rnnMaskDesc, "num_step", numStep);
        ge::AttrUtils::SetInt(rnnMaskDesc, "hidden_size", hiddenSize);

        // Creat Mask
        ge::NodePtr maskNode = graph.AddNode(rnnMaskDesc);
        FUSION_PASS_CHECK(maskNode == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Create Mask node:%s failed", rnnMaskDesc->GetName().c_str()),
            return FAILED);
        newNodes.push_back(maskNode);

        // Add Edge
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                               maskNode->GetInDataAnchor(0)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add Mask input edge failed"), return FAILED);

        // Add Edge
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(maskNode->GetOutDataAnchor(0), dynamicRnnNode->GetInDataAnchor(3)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);

        return SUCCESS;
    }

Status CommonLSTMFusionPass::AddSliceConcatNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode,
                                                ge::NodePtr dynamicRnnForwardNode, ge::NodePtr dynamicRnnReverseNode,
                                                ge::GeTensorDesc dynamicRnnOutputDesc, vector<ge::NodePtr> &newNodes,
                                                std::string nodeName, int nodeIndex) {
  // forward strided_slice
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string operatorName = fusedDesc->GetName() + "/StridedSliceDForward" + nodeName;
  auto sliceOpForward = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "StridedSliceD");
  FUSION_PASS_CHECK(sliceOpForward.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op forward Op error"),
                    return FAILED);
  auto slice_desc_forward = ge::OpDescUtils::GetOpDescFromOperator(sliceOpForward);
  sliceOpForward.BreakConnect();

  ge::GeTensorDesc SliceTensorDesc = fusedDesc->GetOutputDesc(nodeIndex).Clone();
  std::vector<int64_t> slice_dims = SliceTensorDesc.GetShape().GetDims();
  slice_dims[0] = 1;
  ge::GeShape slice_shape(slice_dims);
  SliceTensorDesc.SetShape(slice_shape);
  SliceTensorDesc.SetOriginShape(slice_shape);

  slice_desc_forward->UpdateInputDesc("x", dynamicRnnOutputDesc);
  slice_desc_forward->UpdateOutputDesc("y", SliceTensorDesc);
  ge::AttrUtils::SetListInt(slice_desc_forward, "begin", {0, 0, 0});
  ge::AttrUtils::SetListInt(slice_desc_forward, "end", {1, slice_dims[1], slice_dims[2]});
  ge::AttrUtils::SetListInt(slice_desc_forward, "strides", {1, 1, 1});

  ge::NodePtr slice_node_forward = graph.AddNode(slice_desc_forward);
  newNodes.push_back(slice_node_forward);

  // reverse strided_slice
  operatorName = fusedDesc->GetName() + "/StridedSliceDReverse" + nodeName;
  auto sliceOpReverse = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "StridedSliceD");
  FUSION_PASS_CHECK(sliceOpReverse.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op reverse Op error"),
                    return FAILED);
  auto slice_desc_reverse = ge::OpDescUtils::GetOpDescFromOperator(sliceOpReverse);
  sliceOpReverse.BreakConnect();

  slice_desc_reverse->UpdateInputDesc("x", dynamicRnnOutputDesc);
  slice_desc_reverse->UpdateOutputDesc("y", SliceTensorDesc);
  ge::AttrUtils::SetListInt(slice_desc_reverse, "begin", {0, 0, 0});
  ge::AttrUtils::SetListInt(slice_desc_reverse, "end", {1, slice_dims[1], slice_dims[2]});
  ge::AttrUtils::SetListInt(slice_desc_reverse, "strides", {1, 1, 1});

  ge::NodePtr slice_node_reverse = graph.AddNode(slice_desc_reverse);
  newNodes.push_back(slice_node_reverse);

  // connect dynamicRnn output hc -> slice
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnForwardNode->GetOutDataAnchor(nodeIndex),
                                         slice_node_forward->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicrnn edge to fusion slice_node failed."), return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnReverseNode->GetOutDataAnchor(nodeIndex),
                                         slice_node_reverse->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicrnn edge to fusion slice_node failed."), return FAILED);

  // create dynamicRnn output concat node
  auto concatOp = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/ConcatD_" + nodeName, "ConcatD");
  FUSION_PASS_CHECK(concatOp.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create concat %s operator error", nodeName.c_str()),
                    return FAILED);
  auto concat_desc = ge::OpDescUtils::GetOpDescFromOperator(concatOp);
  concatOp.BreakConnect();

  ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(nodeIndex);
  concat_desc->AddInputDesc("x_forward", SliceTensorDesc);
  concat_desc->AddInputDesc("x_reverse", SliceTensorDesc);
  concat_desc->UpdateOutputDesc("y", originTensorDesc);

  ge::AttrUtils::SetInt(concat_desc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concat_desc, "N", 2);

  ge::NodePtr concat_node = graph.AddNode(concat_desc);
  newNodes.push_back(concat_node);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(slice_node_forward->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add silce forward edge to fusion slice_node failed."), return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(slice_node_reverse->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice reverse edge to fusion slice_node failed."), return FAILED);

  for (InDataAnchorPtr oriTopPeerAnchorPtri : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    oriTopPeerAnchorPtri->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add concat node to fusion node output failed."), return FAILED);
  }
  return SUCCESS;
}

Status CommonLSTMFusionPass::AddSliceNode(ge::ComputeGraph &graph, ge::NodePtr fusedNode,
                                          ge::NodePtr dynamicRnnNode, ge::GeTensorDesc dynamicRnnOutputDesc,
                                          vector<ge::NodePtr> &newNodes, std::string nodeName, int nodeIndex) {
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  std::string operatorName = fusedDesc->GetName() + "/StridedSliceD" + nodeName;
  auto sliceOp = ge::OperatorFactory::CreateOperator(operatorName.c_str(), "StridedSliceD");
  FUSION_PASS_CHECK(sliceOp.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice_op Op error"),
                    return FAILED);
  auto sliceDesc = ge::OpDescUtils::GetOpDescFromOperator(sliceOp);
  sliceOp.BreakConnect();

  ge::GeTensorDesc SliceTensorDesc = fusedDesc->GetOutputDesc(nodeIndex).Clone();
  std::vector<int64_t> SliceDims = SliceTensorDesc.GetShape().GetDims();
  SliceDims[0] = 1;
  ge::GeShape SliceShape(SliceDims);
  SliceTensorDesc.SetShape(SliceShape);
  SliceTensorDesc.SetOriginShape(SliceShape);

  sliceDesc->UpdateInputDesc("x", dynamicRnnOutputDesc);
  sliceDesc->UpdateOutputDesc("y", SliceTensorDesc);
  ge::AttrUtils::SetListInt(sliceDesc, "begin", {-1, 0, 0});
  ge::AttrUtils::SetListInt(sliceDesc, "end", {-2, SliceDims[1], SliceDims[2]});
  ge::AttrUtils::SetListInt(sliceDesc, "strides", {-1, 1, 1});

  ge::NodePtr SliceNode = graph.AddNode(sliceDesc);
  newNodes.push_back(SliceNode);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnNode->GetOutDataAnchor(nodeIndex),
                                         SliceNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add dynamicrnn edge to fusion slice_node failed."),
      return FAILED);

  for (InDataAnchorPtr oriTopPeerAnchorPtri : fusedNode->GetOutDataAnchor(nodeIndex)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(nodeIndex), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice node to fusion node output failed."),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(SliceNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add slice node to fusion node output failed."),
        return FAILED);
  }
  return SUCCESS;
}

ge::OpDescPtr CommonLSTMFusionPass::CreateSplitDesc(ge::OpDescPtr splitDesc, ge::OpDescPtr fusedDesc,
                                                    string tensorName, int64_t splitDim) {
  ge::GeTensorDesc tensorDesc = fusedDesc->GetInputDesc(tensorName).Clone();
  splitDesc->AddInputDesc(tensorDesc);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", splitDim);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);
  vector<int64_t> tensorDims = tensorDesc.GetShape().GetDims();

  tensorDims[splitDim] = 1;

  ge::GeShape tensorShape(tensorDims);
  tensorDesc.SetShape(tensorShape);
  tensorDesc.SetOriginShape(tensorShape);

  splitDesc->AddOutputDesc(tensorDesc);
  splitDesc->AddOutputDesc(tensorDesc);

  return splitDesc;
}

Status CommonLSTMFusionPass::SetOutputTensorDescAttr(uint16_t originOutputIndex, uint16_t fuseOutputIndex,
                                                     ge::NodePtr originNode, ge::NodePtr fuseNode) {
  FUSION_PASS_CHECK(fuseNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode is null"), return FAILED);
  FUSION_PASS_CHECK(originNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "originNode is null"), return FAILED);
  FUSION_PASS_CHECK(fuseNode->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode OpDesc is null"),
                    return FAILED);
  FUSION_PASS_CHECK(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex) == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "fuseNode outputDesc is null"), return FAILED);
  FUSION_PASS_CHECK(originNode->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "originNode OpDesc is null"),
                    return FAILED);
  ge::AttrUtils::SetStr(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME,
                        originNode->GetName());
  ge::AttrUtils::SetInt(fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex),
                        ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, originOutputIndex);
  GraphPassUtil::SetDataDumpOriginDataType(
      originNode->GetOpDesc()->GetOutputDesc(originOutputIndex).GetOriginDataType(),
      fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex));
  GraphPassUtil::SetDataDumpOriginFormat(originNode->GetOpDesc()->GetOutputDesc(originOutputIndex).GetOriginFormat(),
                                         fuseNode->GetOpDesc()->MutableOutputDesc(fuseOutputIndex));
  return SUCCESS;
}

Status CommonLSTMFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of LSTM
  OP_LOGI(FUSED_OP_TYPE.c_str(), "common lstm start fusion");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "fusedNode is null, fusion failed."),
  return PARAM_INVALID);

  // get the OpDescPtr of LSTM
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "fusedNode OpDesc is null, fusion failed."),
  return PARAM_INVALID);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);

  std::string direction;
  std::map<string, string> directionMapInfo { {"forward", "UNIDIRECTIONAL"}, {"reverse", "REDIRECTIONAL"},
                                              {"bidirectional", "BIDIRECITIONAL"}};
  ge::AttrUtils::GetStr(fusedDesc, "direction", direction);

  std::vector<string> singleList {"forward", "reverse"};
  bool is_single = std::find(singleList.begin(), singleList.end(), direction) != singleList.end();
  if (direction == "bidirectional") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "bidirectional lstm start fusion.");
    int32_t num_directions = 2;
    auto dynamicRnnOpForward = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/DynamicRNN" + "Forward", "DynamicRNN");
    FUSION_PASS_CHECK(dynamicRnnOpForward.IsEmpty(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicRnn forward operator error"),
    return FAILED);
    auto dynamicRnnDescForward = ge::OpDescUtils::GetOpDescFromOperator(dynamicRnnOpForward);
    dynamicRnnOpForward.BreakConnect();

    auto dynamicRnnOpReverse = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/DynamicRNN" + "Reverse", "DynamicRNN");
    FUSION_PASS_CHECK(dynamicRnnOpReverse.IsEmpty(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicRnn reverse operator error"),
    return FAILED);
    auto dynamicRnnDescReverse = ge::OpDescUtils::GetOpDescFromOperator(dynamicRnnOpReverse);
    dynamicRnnOpReverse.BreakConnect();

    // set dynamicrnn direction
    ge::AttrUtils::SetStr(dynamicRnnDescForward, "direction", directionMapInfo["forward"]);
    ge::AttrUtils::SetStr(dynamicRnnDescReverse, "direction", directionMapInfo["reverse"]);

    // process seq_length
    bool hasSeqLength = fusedDesc->MutableInputDesc("sequence_lens") != nullptr;
    if (hasSeqLength) {
      ge::GeTensorDesc seq_length_desc = *fusedDesc->MutableInputDesc("sequence_lens");
      dynamicRnnDescForward->UpdateInputDesc("seq_length", seq_length_desc);
      dynamicRnnDescReverse->UpdateInputDesc("seq_length", seq_length_desc);
    }

    // process init_h
    InputIndexInfo inputIndexInfo;
    bool hasInitH = fusedDesc->MutableInputDesc("initial_h") != nullptr;
    std::vector<ge::GeTensorPtr> initHPtrs;
    ge::NodePtr splitHNode = nullptr;
    bool initHConst = true;
    if (hasInitH) {
      ge::OutDataAnchorPtr initHAnchorPtr = fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor();
      ge::NodePtr initHNode = initHAnchorPtr->GetOwnerNode();
      string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(initHNode);
      if (nodeType != "Const" && nodeType != "Constant") {
        initHConst = false;
        ge::GeTensorDesc initTensorDesc = fusedDesc->GetInputDesc("initial_h").Clone();
        vector<int64_t> initTensorDims = initTensorDesc.GetShape().GetDims();
        initTensorDims[0] = 1;
        ge::GeShape initTensorShape(initTensorDims);
        initTensorDesc.SetShape(initTensorShape);
        initTensorDesc.SetOriginShape(initTensorShape);

        ge::OpDescPtr splitHDesc = nullptr;
        FUSION_PASS_MAKE_SHARED(
          (splitHDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/splitH", "SplitD")),
          return INTERNAL_ERROR);
        splitHDesc = CreateSplitDesc(splitHDesc, fusedDesc, "initial_h", 0);
        dynamicRnnDescForward->AddInputDesc("init_h", initTensorDesc);
        dynamicRnnDescReverse->AddInputDesc("init_h", initTensorDesc);

        splitHNode = graph.AddNode(splitHDesc);
        newNodes.push_back(splitHNode);
      } else {
        Status retInitH = ProcessLSTMInitH(fusedNode, inputIndexInfo, initHPtrs);
        FUSION_PASS_CHECK(retInitH != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process init_h fail."), return FAILED);
        dynamicRnnDescForward->UpdateInputDesc("init_h", initHPtrs[0]->GetTensorDesc());
        dynamicRnnDescReverse->UpdateInputDesc("init_h", initHPtrs[1]->GetTensorDesc());
      }
    }

    // process init_c
    bool hasInitC = fusedDesc->MutableInputDesc("initial_c") != nullptr;
    std::vector<ge::GeTensorPtr> initCPtrs;
    ge::NodePtr splitCNode = nullptr;
    bool initCConst = true;
    if (hasInitC) {
      ge::OutDataAnchorPtr initCAnchorPtr = fusedNode->GetInDataAnchor(6)->GetPeerOutAnchor();
      ge::NodePtr initCNode = initCAnchorPtr->GetOwnerNode();
      string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(initCNode);
      if (nodeType != "Const" && nodeType != "Constant") {
        initCConst = false;
        ge::GeTensorDesc initTensorDesc = fusedDesc->GetInputDesc("initial_c").Clone();
        vector<int64_t> initTensorDims = initTensorDesc.GetShape().GetDims();
        initTensorDims[0] = 1;
        ge::GeShape initTensorShape(initTensorDims);
        initTensorDesc.SetShape(initTensorShape);
        initTensorDesc.SetOriginShape(initTensorShape);

        ge::OpDescPtr splitCDesc = nullptr;
        FUSION_PASS_MAKE_SHARED(
          (splitCDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/splitC", "SplitD")),
          return INTERNAL_ERROR);
        splitCDesc = CreateSplitDesc(splitCDesc, fusedDesc, "initial_c", 0);

        dynamicRnnDescForward->AddInputDesc("init_c", initTensorDesc);
        dynamicRnnDescReverse->AddInputDesc("init_c", initTensorDesc);

        splitCNode = graph.AddNode(splitCDesc);
        newNodes.push_back(splitCNode);
      } else {
        Status retInitC = ProcessLSTMInitC(fusedNode, inputIndexInfo, initCPtrs);
        FUSION_PASS_CHECK(retInitC != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process init_c fail."), return FAILED);
        dynamicRnnDescForward->UpdateInputDesc("init_c", initCPtrs[0]->GetTensorDesc());
        dynamicRnnDescReverse->UpdateInputDesc("init_c", initCPtrs[1]->GetTensorDesc());
      }
    }

    // process w
    int32_t hiddenSize = 0;
    std::vector<ge::GeTensorPtr> wTensorPtrs;
    Status retW = ProcessLSTMWxh(fusedNode, inputIndexInfo, hiddenSize, num_directions, wTensorPtrs);
    FUSION_PASS_CHECK(retW != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
    dynamicRnnDescForward->UpdateInputDesc("w", wTensorPtrs[0]->GetTensorDesc());
    dynamicRnnDescReverse->UpdateInputDesc("w", wTensorPtrs[1]->GetTensorDesc());

    // process bias
    std::vector<ge::GeTensorPtr> biasTensorPtrs;
    Status retBias = ProcessLSTMBias(fusedNode, inputIndexInfo, num_directions, hiddenSize, biasTensorPtrs);
    FUSION_PASS_CHECK(retBias != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process bias fail."), return FAILED);

    dynamicRnnDescForward->AddInputDesc("b", biasTensorPtrs[0]->GetTensorDesc());
    dynamicRnnDescReverse->AddInputDesc("b", biasTensorPtrs[1]->GetTensorDesc());

    // process x
    ge::GeTensorDesc xDesc = fusedDesc->GetInputDesc(0).Clone();
    vector<int64_t> tensorXDims = xDesc.GetShape().GetDims();
    int64_t inputSize = fusedDesc->GetInputDesc(1).GetShape().GetDim(2);
    tensorXDims[2] = inputSize;
    ge::GeShape tensorXShape(tensorXDims);
    xDesc.SetShape(tensorXShape);
    xDesc.SetOriginShape(tensorXShape);
    dynamicRnnDescForward->UpdateInputDesc("x", xDesc);
    dynamicRnnDescReverse->UpdateInputDesc("x", xDesc);

    // create dynamic_rnn node
    ge::NodePtr dynamicRnnForwardNode = graph.AddNode(dynamicRnnDescForward);
    FUSION_PASS_CHECK(dynamicRnnForwardNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dynamicRnn forward node is null, fusion failed."),
    return FAILED);
    newNodes.push_back(dynamicRnnForwardNode);

    ge::NodePtr dynamicRnnReverseNode = graph.AddNode(dynamicRnnDescReverse);
    FUSION_PASS_CHECK(dynamicRnnReverseNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dynamicRnn forward node is null, fusion failed."),
    return FAILED);
    newNodes.push_back(dynamicRnnReverseNode);

    // connect x
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         dynamicRnnForwardNode->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x edge to fusion node x failed."),
    return FAILED);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         dynamicRnnReverseNode->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add x edge to fusion node x failed."),
    return FAILED);

    // connect w
    ge::OpDescPtr wDescForward = ge::OpDescUtils::CreateConstOp(wTensorPtrs[0]);
    ge::NodePtr wForward_node = graph.AddNode(wDescForward);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(wForward_node->GetOutDataAnchor(0),
                                                         dynamicRnnForwardNode->GetInDataAnchor(1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node w failed."),
    return FAILED);

    ge::OpDescPtr wDescReverse = ge::OpDescUtils::CreateConstOp(wTensorPtrs[1]);
    ge::NodePtr wReverse_node = graph.AddNode(wDescReverse);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(wReverse_node->GetOutDataAnchor(0),
                                                         dynamicRnnReverseNode->GetInDataAnchor(1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node w failed."),
    return FAILED);

    // connect bias
    ge::OpDescPtr biasDescForward = ge::OpDescUtils::CreateConstOp(biasTensorPtrs[0]);
    ge::NodePtr biasForward_node = graph.AddNode(biasDescForward);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(biasForward_node->GetOutDataAnchor(0),
                                                         dynamicRnnForwardNode->GetInDataAnchor(2)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node bias failed."),
    return FAILED);

    ge::OpDescPtr biasDescReverse = ge::OpDescUtils::CreateConstOp(biasTensorPtrs[1]);
    ge::NodePtr biasReverse_node = graph.AddNode(biasDescReverse);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(biasReverse_node->GetOutDataAnchor(0),
                                                         dynamicRnnReverseNode->GetInDataAnchor(2)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node bias failed."),
    return FAILED);

    // connect seq_length
    if (hasSeqLength) {
      FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, dynamicRnnForwardNode, graph, hiddenSize, newNodes),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                        return FAILED);
      FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, dynamicRnnReverseNode, graph, hiddenSize, newNodes),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                        return FAILED);
    }

    // connect init_h
    if (hasInitH) {
      if (initHConst) {
        ge::OpDescPtr initHDescForward = ge::OpDescUtils::CreateConstOp(initHPtrs[0]);
        ge::NodePtr inith_forward_node = graph.AddNode(initHDescForward);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inith_forward_node->GetOutDataAnchor(0),
                                                            dynamicRnnForwardNode->GetInDataAnchor(4)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node init_h failed."),
        return FAILED);
        ge::OpDescPtr initHDescReverse = ge::OpDescUtils::CreateConstOp(initHPtrs[1]);
        ge::NodePtr inith_reverse_node = graph.AddNode(initHDescReverse);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inith_reverse_node->GetOutDataAnchor(0),
                                                            dynamicRnnReverseNode->GetInDataAnchor(4)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node init_h failed."),
        return FAILED);
      } else {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                                           splitHNode->GetInDataAnchor(0)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h edge to fusion node split h failed."),
        return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitHNode->GetOutDataAnchor(0),
                                                            dynamicRnnForwardNode->GetInDataAnchor(4)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node init_h failed."),
        return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitHNode->GetOutDataAnchor(1),
                                                             dynamicRnnReverseNode->GetInDataAnchor(4)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node init_h failed."),
        return FAILED);
      }
    }

    // connect init_c
    if (hasInitC) {
      if (initCConst) {
        ge::OpDescPtr initCDescForward = ge::OpDescUtils::CreateConstOp(initCPtrs[0]);
        ge::NodePtr initc_forward_node = graph.AddNode(initCDescForward);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(initc_forward_node->GetOutDataAnchor(0),
                                                            dynamicRnnForwardNode->GetInDataAnchor(5)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node init_c failed."),
        return FAILED);
        ge::OpDescPtr initCDescReverse = ge::OpDescUtils::CreateConstOp(initCPtrs[1]);
        ge::NodePtr initc_reverse_node = graph.AddNode(initCDescReverse);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(initc_reverse_node->GetOutDataAnchor(0),
                                                            dynamicRnnReverseNode->GetInDataAnchor(5)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node init_c failed."),
        return FAILED);
      } else {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                                                             splitCNode->GetInDataAnchor(0)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add init_h edge to fusion node split c failed."),
        return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitCNode->GetOutDataAnchor(0),
                                                             dynamicRnnForwardNode->GetInDataAnchor(5)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add forward dynamicRnn edge to fusion node init_c failed."),
        return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitCNode->GetOutDataAnchor(1),
                                                             dynamicRnnReverseNode->GetInDataAnchor(5)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reverse dynamicRnn edge to fusion node init_c failed."),
        return FAILED);
      }
    }

    ge::GeTensorDesc outputYDesc = fusedDesc->GetOutputDesc(0).Clone();
    std::vector<int64_t> dims = outputYDesc.GetShape().GetDims();

    dims.erase(std::begin(dims) + 1);
    ge::GeShape y_shape(dims);
    outputYDesc.SetShape(y_shape);
    outputYDesc.SetOriginShape(y_shape);

    // use common rnn y update all output default value
    dynamicRnnDescForward->UpdateOutputDesc("y", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("output_h", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("output_c", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("i", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("j", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("f", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("o", outputYDesc);
    dynamicRnnDescForward->UpdateOutputDesc("tanhc", outputYDesc);

    dynamicRnnDescReverse->UpdateOutputDesc("y", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("output_h", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("output_c", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("i", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("j", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("f", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("o", outputYDesc);
    dynamicRnnDescReverse->UpdateOutputDesc("tanhc", outputYDesc);

    // create x forward reshape node
    auto reshapeOpForward = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/ReshapeForward", "Reshape");
    FUSION_PASS_CHECK(reshapeOpForward.IsEmpty(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create reshape op forward Op operator error"),
                      return FAILED);
    auto reshape_desc_forward = ge::OpDescUtils::GetOpDescFromOperator(reshapeOpForward);
    reshapeOpForward.BreakConnect();

    ge::GeTensorDesc ReshapeTensorDesc = fusedDesc->GetOutputDesc(0).Clone();
    std::vector<int64_t> reshape_dims = ReshapeTensorDesc.GetShape().GetDims();

    reshape_dims[1] = 1;
    ge::GeShape r_shape(reshape_dims);
    ReshapeTensorDesc.SetShape(r_shape);
    ReshapeTensorDesc.SetOriginShape(r_shape);

    reshape_desc_forward->UpdateInputDesc("x", outputYDesc);
    reshape_desc_forward->UpdateInputDesc("shape", ReshapeTensorDesc);
    reshape_desc_forward->UpdateOutputDesc("y", ReshapeTensorDesc);

    ge::NodePtr Reshape_node_forward = graph.AddNode(reshape_desc_forward);
    FUSION_PASS_CHECK(Reshape_node_forward == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape forward node node is null, fusion failed."), return FAILED);
    newNodes.push_back(Reshape_node_forward);

    // create x reverse reshape node
    auto reshapeOpReverse = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/ReshapeReverse", "Reshape");
    FUSION_PASS_CHECK(reshapeOpReverse.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create reverse reshape node error"),
                      return FAILED);
    auto reshape_desc_reverse = ge::OpDescUtils::GetOpDescFromOperator(reshapeOpReverse);
    reshapeOpReverse.BreakConnect();

    reshape_desc_reverse->UpdateInputDesc("x", outputYDesc);
    reshape_desc_reverse->UpdateInputDesc("shape", ReshapeTensorDesc);
    reshape_desc_reverse->UpdateOutputDesc("y", ReshapeTensorDesc);

    ge::NodePtr Reshape_node_reverse = graph.AddNode(reshape_desc_reverse);
    FUSION_PASS_CHECK(Reshape_node_reverse == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape reverse node is null, fusion failed."), return FAILED);
    newNodes.push_back(Reshape_node_reverse);

    // connect y to reshape
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnForwardNode->GetOutDataAnchor(0),
                                                          Reshape_node_forward->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion Reshape_node_forward y failed."),
                      return FAILED);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRnnReverseNode->GetOutDataAnchor(0),
                                                         Reshape_node_reverse->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion Reshape_node_reverse y failed."),
                      return FAILED);

    // create y concat node
    auto concatOpY = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/ConcatD_y", "ConcatD");
    FUSION_PASS_CHECK(concatOpY.IsEmpty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create y concat operator error"),
                      return FAILED);
    auto concatDescY = ge::OpDescUtils::GetOpDescFromOperator(concatOpY);
    concatOpY.BreakConnect();

    ge::GeTensorDesc originTensorDesc = fusedDesc->GetOutputDesc(0);
    concatDescY->AddInputDesc("x_forward", ReshapeTensorDesc);
    concatDescY->AddInputDesc("x_reverse", ReshapeTensorDesc);
    concatDescY->UpdateOutputDesc("y", originTensorDesc);

    ge::AttrUtils::SetInt(concatDescY, "concat_dim", 1);
    ge::AttrUtils::SetInt(concatDescY, "N", 2);

    ge::NodePtr concatY_node = graph.AddNode(concatDescY);
    newNodes.push_back(concatY_node);

    // concat reshape y to concat
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(Reshape_node_forward->GetOutDataAnchor(0),
                                           concatY_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reshape forward edge to fusion concat failed."), return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(Reshape_node_reverse->GetOutDataAnchor(0),
                                           concatY_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add reshape reverse edge to fusion concat failed."), return FAILED);

    ge::OutDataAnchorPtr outputY = fusedNode->GetOutDataAnchor(0);
    auto hOriTopPeerAnchors = outputY->GetPeerInDataAnchors();

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

    for (uint64_t i = 0; i < hOriTopPeerAnchors.size(); i++) {
      ge::InDataAnchorPtr oriTopPeerAnchorPtri = hOriTopPeerAnchors.at(i);
      ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(concatY_node->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add concat Node edge to fusion node output failed."), return FAILED);
    }
    Status retSetDumpData = SetOutputTensorDescAttr(0, 0, fusedNode, dynamicRnnForwardNode);
    if (retSetDumpData == FAILED) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "set forward dump origin data failed!");
    }
    retSetDumpData = SetOutputTensorDescAttr(0, 0, fusedNode, dynamicRnnReverseNode);
    if (retSetDumpData == FAILED) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "set reverse dump origin data failed!");
    }
    FUSION_PASS_CHECK(AddSliceConcatNode(graph, fusedNode, dynamicRnnForwardNode, dynamicRnnReverseNode,
                                         outputYDesc, newNodes, "H", 1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice H Op operator error"), return FAILED);
    FUSION_PASS_CHECK(AddSliceConcatNode(graph, fusedNode, dynamicRnnForwardNode, dynamicRnnReverseNode,
                                         outputYDesc, newNodes, "C", 2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create slice C Op operator error"), return FAILED);
  } else if (is_single) {
    int32_t num_directions = 1;
    auto dynamicRnnOp = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "/DynamicRnn", "DynamicRNN");
    FUSION_PASS_CHECK(dynamicRnnOp.IsEmpty(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicRnn operator error"),
    return FAILED);
    auto dynamicRnnDesc = ge::OpDescUtils::GetOpDescFromOperator(dynamicRnnOp);
    dynamicRnnOp.BreakConnect();

    // process x
    ge::GeTensorDesc xDesc = fusedDesc->GetInputDesc(0).Clone();
    vector<int64_t> tensorXDims = xDesc.GetShape().GetDims();
    int64_t inputSize = fusedDesc->GetInputDesc(1).GetShape().GetDim(2);
    tensorXDims[2] = inputSize;
    ge::GeShape tensorXShape(tensorXDims);
    xDesc.SetShape(tensorXShape);
    xDesc.SetOriginShape(tensorXShape);
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

    // Set Attr
    ge::AttrUtils::SetStr(dynamicRnnDesc, "direction", directionMapInfo[direction]);

    // process w
    InputIndexInfo inputIndexInfo;
    int32_t hiddenSize = 0;
    std::vector<ge::GeTensorPtr> wTensorPtrs;
    Status retW = ProcessLSTMWxh(fusedNode, inputIndexInfo, hiddenSize, num_directions, wTensorPtrs);
    FUSION_PASS_CHECK(retW != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process w fail."), return FAILED);
    dynamicRnnDesc->UpdateInputDesc("w", wTensorPtrs[0]->GetTensorDesc());

    // process bias
    std::vector<ge::GeTensorPtr> biasTensorPtrs;
    Status retBias = ProcessLSTMBias(fusedNode, inputIndexInfo, num_directions, hiddenSize, biasTensorPtrs);
    FUSION_PASS_CHECK(retBias != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process b fail."), return FAILED);
    dynamicRnnDesc->AddInputDesc("b", biasTensorPtrs[0]->GetTensorDesc());

    // create dynamic_rnn node
    ge::NodePtr dynamicRnnNode = graph.AddNode(dynamicRnnDesc);
    FUSION_PASS_CHECK(dynamicRnnNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                              "dynamicLSTM node is null, fusion failed."),
    return FAILED);
    newNodes.push_back(dynamicRnnNode);

    // connect x
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         dynamicRnnNode->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node x failed."),
    return FAILED);

    // connect w
    ge::OpDescPtr wDesc = ge::OpDescUtils::CreateConstOp(wTensorPtrs[0]);
    ge::NodePtr wNode = graph.AddNode(wDesc);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(wNode->GetOutDataAnchor(0),
                                           dynamicRnnNode->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node w failed."),
    return FAILED);

    // connect bias
    ge::OpDescPtr biasDesc = ge::OpDescUtils::CreateConstOp(biasTensorPtrs[0]);
    ge::NodePtr biasNode = graph.AddNode(biasDesc);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(biasNode->GetOutDataAnchor(0),
                                           dynamicRnnNode->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node bias failed."),
    return FAILED);

    // connect seq_length
    if (hasSeqLength) {
      FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, dynamicRnnNode, graph, hiddenSize, newNodes),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                        return FAILED);
    }

    // connect init_h
    if (hasInitH) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                             dynamicRnnNode->GetInDataAnchor(4)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node bias failed."),
      return FAILED);
    }

    // connect init_c
    if (hasInitC) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                                             dynamicRnnNode->GetInDataAnchor(5)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add DynamicRNN edge to fusion node bias failed."),
      return FAILED);
    }

    // use common rnn output y
    ge::GeTensorDesc outputYDesc = fusedDesc->GetOutputDesc(0).Clone();
    std::vector<int64_t> dims = outputYDesc.GetShape().GetDims();

    // t derec batch hiden
    dims.erase(std::begin(dims) + 1);
    ge::GeShape y_shape(dims);
    outputYDesc.SetShape(y_shape);
    outputYDesc.SetOriginShape(y_shape);

    // use common rnn y update all output default value
    dynamicRnnDesc->UpdateOutputDesc("y", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("output_h", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("output_c", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("i", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("j", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("f", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("o", outputYDesc);
    dynamicRnnDesc->UpdateOutputDesc("tanhc", outputYDesc);

    FUSION_PASS_CHECK(AddExpandDimsNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "Y", 0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDimsY Op operator error"), return FAILED);

    FUSION_PASS_CHECK(AddSliceNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "H", 1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDimsH Op operator error"), return FAILED);

    FUSION_PASS_CHECK(AddSliceNode(graph, fusedNode, dynamicRnnNode, outputYDesc, newNodes, "C", 2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ExpandDimsC Op operator error"), return FAILED);

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
    Status retSetDumpData = SetOutputTensorDescAttr(0, 0, fusedNode, dynamicRnnNode);
    if (retSetDumpData == FAILED) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "set dump origin data failed!");
    }
  }
  // remove LSTM from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
  return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "common lstm end fusion");
  return SUCCESS;
}

REGISTER_PASS("CommonLSTMFusionPass", BUILT_IN_GRAPH_PASS, CommonLSTMFusionPass);
} // namespace fe
