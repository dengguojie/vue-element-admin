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

/*!
 * \file lstm_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#include "lstm_fusion_pass.h"

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


static const char* FUSED_NODE = "LSTM";
static const std::string PATTERN_FUSEDNODE = "LSTM";

vector<FusionPattern*> ALSTMFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ALSTMFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

string GetPeerNodeOptype(ge::NodePtr fusedNode) {
  ge::InDataAnchorPtr inputXAnchorPtr0 = fusedNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr xAnchorPtr0 = inputXAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputXNode = xAnchorPtr0->GetOwnerNode();
  string opType = inputXNode->GetType();
  return opType;
}

ge::GeTensorDesc ALSTMFusionPass::ProcessStatic(ge::NodePtr fusedNode, int32_t num_output,
                                                ge::NodePtr& innerproductNode, ge::NodePtr& dequantNode,
                                                ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                bool& failStatus, int32_t xStaticIndex, int32_t wxStaticIndex) {
  string opType = GetPeerNodeOptype(fusedNode);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  ge::GeTensorDesc inputTensorDesc = fusedDesc->GetInputDesc(xStaticIndex);
  DataType dataType = inputTensorDesc.GetDataType();
  // create the OpDescPtr for InnerProduct
  ge::OpDescPtr innerProductStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (innerProductStaticDesc =
          std::make_shared<ge::OpDesc>(string("FullyConnection/") + fusedDesc->GetName(), "FullyConnection")),
      failStatus = true; return inputTensorDesc);

  ge::GeShape outputShape = inputTensorDesc.GetShape();
  std::vector<int64_t> dimsInputXShape;
  dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(0));
  dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(1));

  ge::GeShape inputXShape(dimsInputXShape);
  inputTensorDesc.SetShape(inputXShape);
  innerProductStaticDesc->AddInputDesc("x", inputTensorDesc);
  ge::InDataAnchorPtr inputWAnchorPtr0 = fusedNode->GetInDataAnchor(wxStaticIndex);
  ge::OutDataAnchorPtr constAnchorPtr0 = inputWAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWNode = constAnchorPtr0->GetOwnerNode();
  bool isExistSubGraph = false;

  FUSION_PASS_CHECK(inputWNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "weights Wx node is null."),
                    failStatus = true; return inputTensorDesc);
  if (inputWNode->GetType() == "Data") {
    ge::NodePtr parentNode = NodeUtils::GetParentInput(inputWNode);
    isExistSubGraph = true;
    inputWNode = parentNode;
  }
  FUSION_PASS_CHECK(inputWNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "weights Wx node is null."),
                    failStatus = true; return inputTensorDesc);
  Operator constWxStaticOp = OpDescUtils::CreateOperatorFromNode(inputWNode);
  bool constAdjustFlag = false;
  vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(inputWNode);

  FUSION_PASS_CHECK(weights.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "LSTM weights is null, fusion failed."),
                    failStatus = true; return inputTensorDesc);

  ge::GeTensorPtr inputWConstGeTensor = weights[0];
  ge::GeTensorDesc inputWTensorDesc = inputWConstGeTensor->GetTensorDesc();
  constWxStaticOp.GetAttr("const_adjust_flag", constAdjustFlag);
  if (!constAdjustFlag) {
    int32_t c0 = 16;
    if (opType == "AscendQuant") {
      c0 = 32;
      dataType = ge::DT_INT8;
    };
    int32_t wRow = inputWTensorDesc.GetShape().GetDim(1);
    int32_t wCol = inputWTensorDesc.GetShape().GetDim(0);
    int32_t destWRow = (wRow + 15) / 16 * 16;
    // there why
    int32_t destWCol = 4 * ((wCol / 4 + c0 - 1) / c0 * c0);

    std::vector<int64_t> dimsInputWDim;
    // no need padding
    dimsInputWDim.push_back(destWCol);
    dimsInputWDim.push_back(destWRow);

    dimsInputWDim.push_back(1);
    dimsInputWDim.push_back(1);

    ge::GeShape dimsInputWShape(dimsInputWDim);

    inputWTensorDesc.SetShape(dimsInputWShape);
    inputWTensorDesc.SetOriginShape(dimsInputWShape);
    inputWTensorDesc.SetFormat(ge::FORMAT_NCHW);
    inputWTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);

    fusedNode->GetInDataAnchor(wxStaticIndex)
      ->GetPeerOutAnchor()
      ->GetOwnerNode()
      ->GetOpDesc()
      ->UpdateOutputDesc(0, inputWTensorDesc);
    constWxStaticOp.SetAttr("const_adjust_flag", true);

    unique_ptr<float[]> wStaticPaddData(new (std::nothrow) float[destWCol * destWRow]());
    auto retMem = memset_s(wStaticPaddData.get(), destWCol * destWRow, 0, destWCol * destWRow);
    FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to operate const zero memset_s function!"),
                      failStatus = true);
    float* wStaticOriData = (float*)inputWConstGeTensor->GetData().data();
    float* wStaticWeight = wStaticPaddData.get();
    int32_t singleDestCol = ((wCol / 4 + c0 - 1) / c0 * c0);
    for (int i = 0; i < 4; i++){
      for (int j = 0; j < (wCol / 4); j++){
        for (int k = 0; k < wRow; k++){
          *(wStaticWeight + i * singleDestCol * destWRow + j * destWRow + k) =*(wStaticOriData + i * (wCol / 4)*wRow + j * wRow + k);
        }
      }
    }
    inputWConstGeTensor->SetData(reinterpret_cast<uint8_t*>(wStaticPaddData.get()), (destWCol * destWRow) * sizeof(float));
  }
  if (isExistSubGraph) {
    ge::GeTensorDesc wxStaticTensorDesc = inputWConstGeTensor->GetTensorDesc();
    fusedNode->GetInDataAnchor(wxStaticIndex)
             ->GetPeerOutAnchor()
             ->GetOwnerNode()
             ->GetOpDesc()
             ->UpdateOutputDesc(0, wxStaticTensorDesc);
  }
  innerProductStaticDesc->AddInputDesc("w", inputWTensorDesc);
  inputWConstGeTensor->SetTensorDesc(inputWTensorDesc);

  // output todo shape   product output
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_NCHW, dataType);
  std::vector<int64_t> dimsY;
  dimsY.push_back(inputTensorDesc.GetShape().GetDim(0));

  dimsY.push_back(inputWTensorDesc.GetShape().GetDim(0));
  ge::GeShape dimsYShape(dimsY);
  outputTensorDesc.SetShape(dimsYShape);
  outputTensorDesc.SetOriginShape(dimsYShape);
  outputTensorDesc.SetFormat(ge::FORMAT_NCHW);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  innerProductStaticDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetInt(innerProductStaticDesc, "num_output", num_output);
  ge::AttrUtils::SetBool(innerProductStaticDesc, "transpose", false);
  ge::AttrUtils::SetBool(innerProductStaticDesc, "bias_term", false);
  ge::AttrUtils::SetInt(innerProductStaticDesc, "axis", 1);

  // add the sub operators to the graph
  innerproductNode = graph.AddNode(innerProductStaticDesc);
  FUSION_PASS_CHECK(innerproductNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            innerProductStaticDesc->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(innerproductNode);
  if (dataType == ge::DT_INT8) {
    // create the OpDescPtr for AscendDequant
    ge::OpDescPtr dequantStaticDesc1 = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (dequantStaticDesc1 = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/AscendDequant", "AscendDequant")),
        failStatus = true; return outputTensorDesc);
    dequantStaticDesc1->AddInputDesc("x", outputTensorDesc);
    dequantStaticDesc1->AddInputDesc("deq_scale", fusedDesc->GetInputDesc(9));
    outputTensorDesc.SetDataType(ge::DT_INT32);
    dequantStaticDesc1->AddOutputDesc("y", outputTensorDesc);
    bool sqrt_mode_x_static = false;
    ge::AttrUtils::GetBool(fusedDesc, "sqrt_mode_x_static", sqrt_mode_x_static);
    ge::AttrUtils::SetBool(dequantStaticDesc1, "sqrt_mode", sqrt_mode_x_static);
    ge::AttrUtils::SetBool(dequantStaticDesc1, "relu_flag", false);

    // add the sub operators to the graph
    dequantNode = graph.AddNode(dequantStaticDesc1);
    FUSION_PASS_CHECK(
        dequantNode == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", dequantStaticDesc1->GetName().c_str()),
        failStatus = true);
    newNodes.push_back(dequantNode);
  }
  return outputTensorDesc;
}

ge::GeTensorPtr ALSTMFusionPass::ProcessWxh(ge::NodePtr fusedNode, bool& failStatus, int32_t& wxIndex, int32_t& whIndex,
                                            int32_t c0Index) {
  string opType = GetPeerNodeOptype(fusedNode);
  ge::InDataAnchorPtr inputWxAnchorPtr0 = fusedNode->GetInDataAnchor(wxIndex);
  ge::OutDataAnchorPtr constWxAnchorPtr0 = inputWxAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWxNode = constWxAnchorPtr0->GetOwnerNode();

  bool isExistSubGraph = false;
  FUSION_PASS_CHECK(inputWxNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "weights static node is null."),
                    return nullptr);
  if (inputWxNode->GetType() == "Data") {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Process weight data, the weight input is data.");
    ge::NodePtr parentNode = NodeUtils::GetParentInput(inputWxNode);
    FUSION_PASS_CHECK((parentNode == nullptr) || (parentNode->GetType() != "Const"),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "weights static get parent node failed."),
                       return nullptr);
    isExistSubGraph = true;
    inputWxNode = parentNode;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Process weight data, get the parent node.");
  }

  Operator constWxOp = OpDescUtils::CreateOperatorFromNode(inputWxNode);
  bool constAdjustFlag = false;
  vector<ge::GeTensorPtr> weightsWx = ge::OpDescUtils::MutableWeights(inputWxNode);

  FUSION_PASS_CHECK(weightsWx.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "LSTM weights Wx is null, fusion failed"),
                    failStatus = true;
                    return nullptr);

  ge::GeTensorPtr wxTensorPtr = weightsWx[0];
  constWxOp.GetAttr("const_adjust_flag", constAdjustFlag);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Judge the constAdjustFlag is true.");
  if (constAdjustFlag) {
    if (isExistSubGraph) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Process weight data, update the weight data input desc.");
      ge::GeTensorDesc wxhTensorDesc = wxTensorPtr->GetTensorDesc();
      fusedNode->GetInDataAnchor(wxIndex)
               ->GetPeerOutAnchor()
               ->GetOwnerNode()
               ->GetOpDesc()
               ->UpdateOutputDesc(0, wxhTensorDesc);
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "LSTM const_adjust_flag is true, no need adjust again.");
    return wxTensorPtr;
  }
  ge::InDataAnchorPtr inputWhAnchorPtr0 = fusedNode->GetInDataAnchor(whIndex);
  ge::OutDataAnchorPtr constWhAnchorPtr0 = inputWhAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputWhNode = constWhAnchorPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> weightsWh = ge::OpDescUtils::MutableWeights(inputWhNode);
  FUSION_PASS_CHECK(weightsWh.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "LSTM weightsWh is null, fusion failed."),
                    failStatus = true; return nullptr);

  ge::GeTensorPtr whTensorPtr = weightsWh[0];

  ge::GeTensorDesc wxConstTensorDesc = wxTensorPtr->GetTensorDesc();
  ge::GeTensorDesc whConstTensorDesc = whTensorPtr->GetTensorDesc();

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  DataType dataType = fusedDesc->GetInputDesc(c0Index).GetDataType();
  int32_t c0 = 16;
  if (opType == "AscendQuant") {
    c0 = 32;
    dataType = ge::DT_INT8;
  }
  int32_t wxRow = wxConstTensorDesc.GetShape().GetDim(1);

  int32_t destWxRow = (wxRow + c0 - 1) / c0 * c0;

  int32_t whRow = whConstTensorDesc.GetShape().GetDim(1);
  int32_t whCol = whConstTensorDesc.GetShape().GetDim(0);

  int32_t destWhRow = (whRow + c0 - 1) / c0 * c0;
  int32_t destWhCol = (((whCol / 4) + c0 - 1) / c0 * c0) * 4;

  std::vector<int64_t> dimsIn;

  int32_t targetRow = destWxRow + destWhRow;

  dimsIn.push_back(destWhCol);
  dimsIn.push_back(targetRow);

  ge::GeShape wxhShape(dimsIn);
  ge::GeTensorDesc wxhTensorDesc(wxhShape, ge::FORMAT_HWCN, dataType);
  wxhTensorDesc.SetOriginShape(wxhShape);

  wxhTensorDesc.SetFormat(ge::FORMAT_ND);
  wxhTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  fusedNode->GetInDataAnchor(wxIndex)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->UpdateOutputDesc(0,
                                                                                                         wxhTensorDesc);
  wxTensorPtr->SetTensorDesc(wxhTensorDesc);
  ge::GeTensorPtr weightTensor = nullptr;
  if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> wxhPaddData(new (std::nothrow) float[targetRow * destWhCol]());
    float* wxData = (float*)wxTensorPtr->GetData().data();
    float* whData = (float*)whTensorPtr->GetData().data();

    auto retMemWxh = memset_s(wxhPaddData.get(), targetRow * destWhCol, 0, targetRow * destWhCol);
    FUSION_PASS_CHECK(retMemWxh != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM weightsWh get space failed, fusion failed."), failStatus=true);
    float* dstWeight = wxhPaddData.get();
    int32_t oldSingleCol = whCol / 4;
    int32_t singleCol = (oldSingleCol + 15) / 16 * 16;

    int32_t tarSingleWeigh = targetRow * singleCol;
    int32_t oldSingleWeigh = wxRow * oldSingleCol;
    for (int32_t repeat = 0; repeat < 4; repeat++) {
      int32_t dst_burst = tarSingleWeigh * repeat;
      int32_t src_burst = oldSingleWeigh * repeat;
      for (int32_t roundCol = 0; roundCol < oldSingleCol; roundCol++) {
        int32_t dst_burst_row = targetRow * roundCol;
        int32_t src_burst_row = wxRow * roundCol;
        for (int32_t i = 0; i < wxRow; i++) {
          *(dstWeight + dst_burst + dst_burst_row + i) = *(wxData + src_burst + src_burst_row + i);
        }
      }

      // tensor wh
      int32_t dst_burst_b = tarSingleWeigh * repeat;
      int32_t src_burst_b = whRow * oldSingleCol * repeat;
      for (int32_t roundHCol = 0; roundHCol < oldSingleCol; roundHCol++) {
        int32_t dst_burst_row = targetRow * roundHCol + destWxRow;
        int32_t src_burst_row = whRow * roundHCol;
        for (int32_t i = 0; i < whRow; i++) {
          *(dstWeight + dst_burst_b + dst_burst_row + i) = *(whData + src_burst_b + src_burst_row + i);
        }
      }
    }
    FUSION_PASS_MAKE_SHARED((weightTensor = std::make_shared<GeTensor>(wxhTensorDesc,
                                                                       reinterpret_cast<uint8_t*>(wxhPaddData.get()),
                                                                       destWhCol * targetRow * sizeof(float))),
                            weightTensor = nullptr;
                            return weightTensor);
    ge::AttrUtils::SetTensor(inputWxNode->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weightTensor);
    constWxOp.SetAttr("const_adjust_flag", true);

  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, int8), fusion failed.",
            fusedDesc->GetName().c_str());
    failStatus = true;
  }

  return weightTensor;
}

vector<ge::NodePtr> ALSTMFusionPass::ProcessLstmCellV2(ge::NodePtr fusedNode, ge::ComputeGraph& graph,
                                                       ge::GeTensorDesc& outInnerProductTensorDesc,
                                                       const ge::GeTensorDesc& wxhTensorDesc,
                                                       const ge::GeTensorDesc& hTensorDesc,
                                                       vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                       int32_t biasIndex, bool has_static) {
  vector<ge::NodePtr> lstmCellV2Node = {};
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();

  DataType outputDataType = fusedDesc->GetInputDesc(0).GetDataType();
  DataType startType = ge::DT_FLOAT16;
  if (outputDataType == startType) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op[%s]: the dataType of input 0 is float16.", fusedDesc->GetName().c_str());
  } else if (outputDataType == ge::DT_FLOAT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op[%s]: the dataType of input 0 is float32", fusedDesc->GetName().c_str());
  }
  // bias
  ge::InDataAnchorPtr biasPtr0 = fusedNode->GetInDataAnchor(biasIndex);
  ge::OutDataAnchorPtr constBiasPtr0 = biasPtr0->GetPeerOutAnchor();
  ge::NodePtr inputBiasNode = constBiasPtr0->GetOwnerNode();
  vector<ge::GeTensorPtr> bias_data = ge::OpDescUtils::MutableWeights(inputBiasNode);
  if (bias_data.empty()) {
    failStatus = true;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM bias_data is null, fusion failed.");
    return lstmCellV2Node;
  }
  Operator biasOp = OpDescUtils::CreateOperatorFromNode(inputBiasNode);
  bool constAdjustFlag = false;
  biasOp.GetAttr("const_adjust_flag", constAdjustFlag);
  ge::GeTensorDesc biasTensorDesc = fusedDesc->GetInputDesc(biasIndex);
  if (!constAdjustFlag) {
    ge::GeTensorPtr biasTensorPtr = bias_data[0];
    int32_t bias_dim = biasTensorDesc.GetShape().GetDim(0);
    int32_t tar_bias_dim = (((bias_dim / 4) + 15) / 16 * 16) * 4;
    vector<int64_t> biasDims;
    biasDims.push_back(tar_bias_dim);
    GeShape biasShape(biasDims);
    biasTensorDesc.SetShape(biasShape);
    biasTensorDesc.SetOriginShape(biasShape);
    biasTensorPtr->SetTensorDesc(biasTensorDesc);

    fusedNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->UpdateOutputDesc(
      0, biasTensorDesc);
    unique_ptr<float[]> biasPadData(new (std::nothrow) float[tar_bias_dim]());

    auto retMemBias = memset_s(biasPadData.get(), tar_bias_dim, 0, tar_bias_dim);
    FUSION_PASS_CHECK(retMemBias != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LSTM Bias get space failed, fusion failed."), failStatus=true);
    float* dstBias = biasPadData.get();
    float* srcBias = (float*)biasTensorPtr->GetData().data();
    for (int32_t repeat = 0; repeat < 4; repeat++) {
      int32_t dst_burst = tar_bias_dim / 4 * repeat;
      int32_t src_burst = bias_dim / 4 * repeat;
      for (int32_t i = 0; i < (bias_dim / 4); i++) {
        *(dstBias + dst_burst + i) = *(srcBias + src_burst + i);
      }
    }

    biasTensorPtr->SetData(reinterpret_cast<uint8_t*>(biasPadData.get()), (tar_bias_dim) * sizeof(float));
    biasOp.SetAttr("const_adjust_flag", true);
  }

  string opType = GetPeerNodeOptype(fusedNode);
  DataType dataType = biasTensorDesc.GetDataType();
  if (opType == "AscendQuant") {
    dataType = ge::DT_INT8;
  }

  ge::GeTensorDesc inputTensorDesc0 = fusedDesc->GetInputDesc(0);
  auto shape_0 = inputTensorDesc0.GetShape();

  ge::GeTensorDesc outTensorDesc = fusedDesc->GetOutputDesc(0);
  int32_t hiddenSize = outTensorDesc.GetShape().GetDim(2);
  int32_t tarHiddenDim = ((hiddenSize +15) / 16 * 16) * 4;
  vector<int64_t> outputOriDims;
  outputOriDims.push_back(shape_0.GetDim(1));
  outputOriDims.push_back(hiddenSize);
  GeShape outputOriShape(outputOriDims);

  vector<int64_t> inOriDims;
  vector<int64_t> inOriDims2;
  inOriDims.push_back(1);
  inOriDims.push_back(shape_0.GetDim(1));
  inOriDims.push_back(hiddenSize);
  GeShape inputOriShape(inOriDims);

  inOriDims2.push_back(shape_0.GetDim(1));
  inOriDims2.push_back(hiddenSize);
  GeShape inputOriShape2(inOriDims2);

  ge::GeTensorDesc cTensorDesc = ge::GeTensorDesc(hTensorDesc.GetShape(), ge::FORMAT_ND, dataType);
  cTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  cTensorDesc.SetOriginShape(hTensorDesc.GetShape());

  ge::GeTensorDesc hSenTensorDesc = ge::GeTensorDesc(hTensorDesc.GetShape(), ge::FORMAT_ND, dataType);
  hSenTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  hSenTensorDesc.SetOriginShape(hTensorDesc.GetShape());

  ge::TensorUtils::SetRealDimCnt(cTensorDesc, 2);
  ge::TensorUtils::SetRealDimCnt(hSenTensorDesc, 2);

  int32_t num_output = 0;
  ge::AttrUtils::GetInt(fusedDesc, "num_output", num_output);
  vector<int64_t> oriShapeDims1;
  oriShapeDims1.push_back(1);
  oriShapeDims1.push_back(shape_0.GetDim(1));
  oriShapeDims1.push_back(shape_0.GetDim(2));
  GeShape ori_shape_input_1(oriShapeDims1);

  vector<int64_t> shapeDims1;
  shapeDims1.push_back(1);
  shapeDims1.push_back((shape_0.GetDim(2) + 15) / 16);
  shapeDims1.push_back((shape_0.GetDim(1) + 15) / 16);
  shapeDims1.push_back(16);
  shapeDims1.push_back(16);
  GeShape shape_input_1(shapeDims1);
  ge::GeTensorDesc xInputTensorDesc =
      ge::GeTensorDesc(shape_input_1, ge::FORMAT_FRACTAL_NZ, inputTensorDesc0.GetDataType());
  xInputTensorDesc.SetOriginShape(ori_shape_input_1);
  xInputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorDesc inputContTensorDesc = fusedDesc->GetInputDesc(1);
  auto shape_desc = inputContTensorDesc.GetShape();
  vector<int64_t> shapeDims2;
  int32_t cont_dim = (shape_desc.GetDim(1) + 15) / 16 * 16;
  shapeDims2.push_back(cont_dim);
  GeShape shape_input_2(shapeDims2);

  inputContTensorDesc.SetShape(shape_input_2);
  int64_t tSize = fusedDesc->GetInputDesc(0).GetShape().GetDim(0) - 1;
  for (int64_t i = 0; i < fusedDesc->GetInputDesc(0).GetShape().GetDim(0); i++) {
    // create the OpDescPtr for BasicLSTMCellV2
    ge::OpDescPtr basicLSTMDesc2 = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (basicLSTMDesc2 =
             std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/BasicLSTMCellV2" + std::to_string(i),
                                          "BasicLSTMCellV2")),
        failStatus = true; return lstmCellV2Node);

    basicLSTMDesc2->AddInputDesc("x", xInputTensorDesc);

    ge::GeTensorDesc check_input = basicLSTMDesc2->GetInputDesc(0);

    basicLSTMDesc2->AddInputDesc("cont", inputContTensorDesc);
    if (has_static) {
      basicLSTMDesc2->AddInputDesc("w_xc_x_static", outInnerProductTensorDesc);
    }

    bool expose_hidden = false;
    ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);
    if (i == 0 && expose_hidden) {
      ge::GeTensorDesc hTensorInput =
          ge::GeTensorDesc(hTensorDesc.GetShape(), hTensorDesc.GetFormat(), hTensorDesc.GetDataType());
      hTensorInput.SetOriginShape(inputOriShape);
      ge::GeTensorDesc cTensorInput =
          ge::GeTensorDesc(cTensorDesc.GetShape(), cTensorDesc.GetFormat(), cTensorDesc.GetDataType());
      cTensorInput.SetOriginShape(inputOriShape);
      basicLSTMDesc2->AddInputDesc("h", hTensorInput);
      basicLSTMDesc2->AddInputDesc("c", cTensorInput);
    }
    if (i != 0) {
      expose_hidden = true;
      hSenTensorDesc.SetOriginShape(inputOriShape2);
      cTensorDesc.SetOriginShape(inputOriShape2);
      basicLSTMDesc2->AddInputDesc("h", hSenTensorDesc);
      basicLSTMDesc2->AddInputDesc("c", cTensorDesc);
    }

    // w_x concat w_h -> w_xh
    basicLSTMDesc2->AddInputDesc("w_xh", wxhTensorDesc);
    basicLSTMDesc2->AddInputDesc("bias", biasTensorDesc);

    ge::AttrUtils::SetInt(basicLSTMDesc2, "num_output", num_output);

    ge::AttrUtils::SetBool(basicLSTMDesc2, "expose_hidden", expose_hidden);

    if (dataType == ge::DT_INT8) {
      basicLSTMDesc2->AddInputDesc("w_xh_deqscale", fusedDesc->GetInputDesc(9));
      float xh_scale = 0.0;
      int32_t xh_offset = 0;
      int32_t w_xh_offset = 0;
      bool sqrt_mode_xh = false;
      ge::AttrUtils::GetBool(fusedDesc, "sqrt_mode_xh", sqrt_mode_xh);
      ge::AttrUtils::GetFloat(fusedDesc, "xh_scale", xh_scale);
      ge::AttrUtils::GetInt(fusedDesc, "xh_offset", xh_offset);
      ge::AttrUtils::GetInt(fusedDesc, "w_xh_offset", w_xh_offset);

      ge::AttrUtils::SetInt(basicLSTMDesc2, "xh_scale", xh_scale);
      ge::AttrUtils::SetBool(basicLSTMDesc2, "sqrt_mode_xh", sqrt_mode_xh);
      ge::AttrUtils::SetInt(basicLSTMDesc2, "xh_offset", xh_offset);
      ge::AttrUtils::SetInt(basicLSTMDesc2, "w_xh_offset", w_xh_offset);
    }
    std::vector<int64_t> shapeNd = {};
    shapeNd.push_back(shape_desc.GetDim(1));
    shapeNd.push_back(tarHiddenDim / 4);
    ge::GeShape shapeFz(shapeNd);

    if (i == tSize) {
      ge::GeTensorDesc hTensorDescLast = ge::GeTensorDesc(shapeFz, ge::FORMAT_ND, dataType);
      ge::GeTensorDesc cTensorDescLast = ge::GeTensorDesc(shapeFz, ge::FORMAT_ND, dataType);
      hTensorDescLast.SetOriginShape(outputOriShape);
      cTensorDescLast.SetOriginShape(outputOriShape);
      hTensorDescLast.SetOriginFormat(ge::FORMAT_ND);
      cTensorDescLast.SetOriginFormat(ge::FORMAT_ND);
      basicLSTMDesc2->AddOutputDesc("h_t", hTensorDescLast);
      basicLSTMDesc2->AddOutputDesc("c_t", cTensorDescLast);
    } else {
      ge::GeTensorDesc hTensorCur =
          ge::GeTensorDesc(hTensorDesc.GetShape(), hTensorDesc.GetFormat(), hTensorDesc.GetDataType());
      hTensorCur.SetOriginShape(outputOriShape);
      cTensorDesc.SetOriginShape(outputOriShape);
      basicLSTMDesc2->AddOutputDesc("h_t", hTensorCur);
      basicLSTMDesc2->AddOutputDesc("c_t", cTensorDesc);
    }

    // add the sub operators to the graph
    ge::NodePtr basicLSTMNode = graph.AddNode(basicLSTMDesc2);
    FUSION_PASS_CHECK(
        basicLSTMNode == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", basicLSTMDesc2->GetName().c_str()),
        failStatus = true);
    newNodes.push_back(basicLSTMNode);
    lstmCellV2Node.push_back(basicLSTMNode);
  }
  return lstmCellV2Node;
}
Status ALSTMFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of LSTM
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  int32_t input_size = fusedNode->GetInDataNodes().size();
  // get the OpDescPtr of LSTM
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc tempInput0Desc = fusedDesc->GetInputDesc(0);
  ge::GeShape shapeInput0 = tempInput0Desc.GetShape();
  int64_t last_dim_value = shapeInput0.GetDim(2);
  if (shapeInput0.GetDims().size() == 4) {
    if (PatternFusionUtil::IsUnknownShape(shapeInput0.GetDim(2)) ||
        PatternFusionUtil::IsUnknownShape(shapeInput0.GetDim(3))) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ALSTMFusionPass cannot be applied for unknown shape.");
      return FAILED;
    }
    last_dim_value = shapeInput0.GetDim(2) * shapeInput0.GetDim(3);
  }
  std::vector<int64_t> input0ShapeDim0;
  input0ShapeDim0.push_back(shapeInput0.GetDim(0));
  input0ShapeDim0.push_back(shapeInput0.GetDim(1));
  input0ShapeDim0.push_back(last_dim_value);

  GeShape input0ShapeNew(input0ShapeDim0);
  tempInput0Desc.SetShape(input0ShapeNew);
  tempInput0Desc.SetOriginShape(input0ShapeNew);
  fusedDesc->UpdateInputDesc(0, tempInput0Desc);
  // input index
  int32_t xStaticIndex = 0;
  int32_t h0Index = 0;
  int32_t c0Index = 0;

  int32_t wxIndex = 0;
  int32_t biasIndex = 0;
  int32_t whIndex = 0;
  int32_t wxStaticIndex = 0;
  bool has_static = false;

  bool expose_hidden = false;
  ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);

  if (input_size == 9) {
    xStaticIndex = 2;
    h0Index = 3;
    c0Index = 4;
    biasIndex = 6;
    wxIndex = 5;
    whIndex = 8;
    wxStaticIndex = 7;
    has_static = true;
  } else if (input_size == 7 and expose_hidden) {
    xStaticIndex = -1;
    h0Index = 2;
    c0Index = 3;
    biasIndex = 5;
    wxIndex = 4;
    whIndex = 6;
    wxStaticIndex = -1;
  } else if (input_size == 7) {
    biasIndex = 4;
    has_static = true;
    wxIndex = 3;
    whIndex = 6;
    xStaticIndex = 2;
    wxStaticIndex = 5;
    h0Index = -1;
    c0Index = -1;

  } else if (input_size == 5) {
    biasIndex = 3;
    wxIndex = 2;
    whIndex = 4;
  }

  if (!has_static) {
    return SUCCESS;
  }
  ge::GeTensorDesc outTensorDesc = fusedDesc->GetOutputDesc(0);
  int32_t outputDim = outTensorDesc.GetShape().GetDim(2);

  int32_t num_output = 0;
  ge::AttrUtils::GetInt(fusedDesc, "num_output", num_output);
  ge::NodePtr innerproductNode;
  ge::NodePtr dequantNode;
  ge::GeTensorDesc outInnerProductTensorDesc;

  ge::GeTensorDesc wxDesc = fusedDesc->GetInputDesc(wxIndex);
  ge::GeShape wxShape = wxDesc.GetShape();
  ge::GeTensorDesc whDesc = fusedDesc->GetInputDesc(whIndex);
  ge::GeShape whShape = whDesc.GetShape();

  bool failStatus = false;
  if (has_static) {
    outInnerProductTensorDesc = ProcessStatic(fusedNode, num_output, innerproductNode, dequantNode, graph, newNodes,
                                              failStatus, xStaticIndex, wxStaticIndex);
    FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProcessStatic:check failed, fusion failed."),
                      return PARAM_INVALID);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Finash process static data.");
  }
  // create the OpDescPtr for SplitVD x
  ge::GeTensorDesc inputTensorDesc0 = fusedDesc->GetInputDesc(0);
  int32_t numSplitX = inputTensorDesc0.GetShape().GetDim(0);
  DataType dataType = inputTensorDesc0.GetDataType();
  DataType startType = ge::DT_FLOAT16;
  if (dataType == startType) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op[%s]: the dataType of input 0 is float16.", fusedDesc->GetName().c_str());
  } else if (dataType == ge::DT_FLOAT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op[%s]: the dataType of input 0 is float32.", fusedDesc->GetName().c_str());
  }
  ge::GeTensorDesc splitNode1Desc =
      ge::GeTensorDesc(inputTensorDesc0.GetShape(), inputTensorDesc0.GetFormat(), startType);
  splitNode1Desc.SetOriginShape(inputTensorDesc0.GetOriginShape());
  ge::OpDescPtr spiltStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (spiltStaticDesc = std::make_shared<ge::OpDesc>(string("SplitVD1/") + fusedDesc->GetName(), "SplitVD")),
      return PARAM_INVALID);
  spiltStaticDesc->AddInputDesc("input_value", splitNode1Desc);
  ge::GeShape shape0 = inputTensorDesc0.GetShape();
  int64_t tSize = shape0.GetDim(0);
  shape0.SetDim(0, 1);
  inputTensorDesc0.SetShape(shape0);
  inputTensorDesc0.SetOriginShape(shape0);
  int64_t nDim = shape0.GetDim(1);
  std::vector<int64_t> outputDim0;
  outputDim0.push_back(1);
  outputDim0.push_back(nDim);
  outputDim0.push_back(outputDim);
  ge::GeShape outputShape(outputDim0);
  std::vector<int64_t> outputDim2;
  outputDim2.push_back(tSize);
  outputDim2.push_back(nDim);
  outputDim2.push_back(outputDim);
  ge::GeShape output2Shape(outputDim2);

  ge::GeTensorDesc output0TensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  output0TensorDesc.SetOriginShape(outputShape);
  ge::GeTensorDesc output1TensorDesc = ge::GeTensorDesc(output2Shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  output1TensorDesc.SetOriginShape(output2Shape);
  ge::GeTensorDesc output2TensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  output2TensorDesc.SetOriginShape(outputShape);

  for (int64_t i = 0; i < numSplitX; i++) {
    spiltStaticDesc->AddOutputDesc("output_data" + std::to_string(i + 1), inputTensorDesc0);
  }
  std::vector<int64_t> sizeSplitsX = {};
  for (int64_t i = 0; i < numSplitX; i++) {
    sizeSplitsX.push_back(1);
  }
  ge::AttrUtils::SetListInt(spiltStaticDesc, "size_splits", sizeSplitsX);
  ge::AttrUtils::SetInt(spiltStaticDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(spiltStaticDesc, "num_split", numSplitX);
  // add the sub operators to the graph
  ge::NodePtr splitXNode = graph.AddNode(spiltStaticDesc);
  FUSION_PASS_CHECK(
      splitXNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", spiltStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(splitXNode);

  // create the OpDescPtr for SplitVD cont
  ge::GeTensorDesc inputContTensorDesc = fusedDesc->GetInputDesc(1);

  int32_t numSplitCont = inputContTensorDesc.GetShape().GetDim(0);
  ge::OpDescPtr spiltContStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (spiltContStaticDesc = std::make_shared<ge::OpDesc>(string("SplitVD2/") + fusedDesc->GetName(), "SplitVD")),
      return PARAM_INVALID);
  ge::GeTensorDesc splitInputDesc =
      ge::GeTensorDesc(inputContTensorDesc.GetShape(), inputContTensorDesc.GetFormat(), ge::DT_FLOAT16);
  splitInputDesc.SetOriginShape(inputContTensorDesc.GetShape());
  spiltContStaticDesc->AddInputDesc("input_value", splitInputDesc);
  ge::GeShape shape1 = splitInputDesc.GetShape();
  shape1.SetDim(0, 1);
  splitInputDesc.SetShape(shape1);
  splitInputDesc.SetOriginShape(shape1);
  for (int64_t i = 0; i < numSplitCont; i++) {
    spiltContStaticDesc->AddOutputDesc("output_data" + std::to_string(i + 1), splitInputDesc);
  }
  std::vector<int64_t> sizeSplitsCont = {};
  for (int64_t i = 0; i < numSplitCont; i++) {
    sizeSplitsCont.push_back(1);
  }
  ge::AttrUtils::SetListInt(spiltContStaticDesc, "size_splits", sizeSplitsCont);
  ge::AttrUtils::SetInt(spiltContStaticDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(spiltContStaticDesc, "num_split", numSplitCont);

  // add the sub operators to the graph
  ge::NodePtr splitContNode = graph.AddNode(spiltContStaticDesc);
  FUSION_PASS_CHECK(
      splitContNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", spiltContStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(splitContNode);
  // process w_xh
  ge::GeTensorPtr wxTensorPtr = ProcessWxh(fusedNode, failStatus, wxIndex, whIndex, c0Index);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProcessWxh:check failed, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(wxTensorPtr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process wxTensorPtr fail."),
                    return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Finash process weight data.");
  std::vector<int64_t> dimsTShape;
  int32_t hDim1 = (inputTensorDesc0.GetShape().GetDim(1) + 15) / 16 * 16;
  int32_t hDim2 = (outputDim + 15) / 16 * 16;

  dimsTShape.push_back(hDim1);
  dimsTShape.push_back(hDim2);
  ge::GeShape inputCShape(dimsTShape);

  ge::GeTensorDesc hTensorDesc = ge::GeTensorDesc(inputCShape, ge::FORMAT_ND, dataType);
  hTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  hTensorDesc.SetOriginShape(inputCShape);
  ge::TensorUtils::SetRealDimCnt(hTensorDesc, 2);

  // create lstmCellV2 Node
  vector<ge::NodePtr> lstmCellV2Node =
      ProcessLstmCellV2(fusedNode, graph, outInnerProductTensorDesc, wxTensorPtr->GetTensorDesc(), hTensorDesc,
                        newNodes, failStatus, biasIndex, has_static);
  FUSION_PASS_CHECK(
      lstmCellV2Node.empty(),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "basiclstm Op build failed."),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Finash build basiclstm op.");
  // create the OpDescPtr for concat
  ge::OpDescPtr concatStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatStaticDesc = std::make_shared<ge::OpDesc>(string("ConcatD/") + fusedDesc->GetName(), "ConcatD")),
      return FAILED);

  for (int64_t i = 0; i < numSplitX; i++) {
    std::vector<int64_t> tempShape;
    tempShape.push_back(1);
    tempShape.push_back(nDim);
    tempShape.push_back(outputDim);
    ge::GeShape tempInputShape(tempShape);
    ge::GeTensorDesc temphTensorDesc = ge::GeTensorDesc(tempInputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
    temphTensorDesc.SetOriginShape(tempInputShape);
    temphTensorDesc.SetOriginFormat(ge::FORMAT_ND);
    concatStaticDesc->AddInputDesc("input_values" + std::to_string(i + 1), temphTensorDesc);
  }
  std::vector<int64_t> totalDims = {};
  totalDims.push_back(tSize);
  totalDims.push_back(inputTensorDesc0.GetShape().GetDim(1));
  totalDims.push_back(outputDim);
  ge::GeShape totalShape(totalDims);
  ge::GeTensorDesc totalOutput = ge::GeTensorDesc(totalShape, ge::FORMAT_ND, dataType);
  totalOutput.SetOriginFormat(ge::FORMAT_ND);
  totalOutput.SetOriginShape(totalShape);
  concatStaticDesc->AddOutputDesc("output_data", totalOutput);
  ge::AttrUtils::SetInt(concatStaticDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatStaticDesc, "N", numSplitX);
  // add the sub operators to the graph
  ge::NodePtr concatNode = graph.AddNode(concatStaticDesc);
  FUSION_PASS_CHECK(
      concatNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", concatStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(concatNode);
  string opType = GetPeerNodeOptype(fusedNode);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Finash add concat node.");
  // process edge
  if (has_static) {
    // connect input0
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(xStaticIndex)->GetPeerOutAnchor(),
                                           innerproductNode->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0),
        return FAILED);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(wxStaticIndex)->GetPeerOutAnchor(),
                                           innerproductNode->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0);

    if (opType == "AscendQuant") {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(innerproductNode->GetOutDataAnchor(0), dequantNode->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                  innerproductNode->GetName().c_str(), 0, dequantNode->GetName().c_str(), 0),
          return FAILED);

      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(10)->GetPeerOutAnchor(),
                                                           dequantNode->GetInDataAnchor(1)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                fusedNode->GetName().c_str(), 0, dequantNode->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              fusedNode->GetName().c_str(), 0, dequantNode->GetName().c_str(), 0);
    }
  }
  FUSION_PASS_CHECK(
      SUCCESS !=
          ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(), splitXNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              fusedNode->GetName().c_str(), 0, splitXNode->GetName().c_str(), 0),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
          fusedNode->GetName().c_str(), 0, splitXNode->GetName().c_str(), 0);

  FUSION_PASS_CHECK(
      SUCCESS !=
          ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(), splitContNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              fusedNode->GetName().c_str(), 1, splitContNode->GetName().c_str(), 0),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
          fusedNode->GetName().c_str(), 1, splitContNode->GetName().c_str(), 0);

  // connect lstmcellv2
  for (int64_t i = 0; i < numSplitX; i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(splitXNode->GetOutDataAnchor(i), lstmCellV2Node[i]->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                splitXNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            splitXNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(splitContNode->GetOutDataAnchor(i), lstmCellV2Node[i]->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%ld] to fusion node:%s's input[%d] failed.",
                splitContNode->GetName().c_str(), i, lstmCellV2Node[i]->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            splitContNode->GetName().c_str(), i, lstmCellV2Node[i]->GetName().c_str(), 0);

    int index = 2;
    if (has_static) {
      if (opType == "AscendQuant") {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dequantNode->GetOutDataAnchor(0),
                                                             lstmCellV2Node[i]->GetInDataAnchor(index)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                  dequantNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                          return FAILED);
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
                dequantNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);
      } else {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(innerproductNode->GetOutDataAnchor(0),
                                                             lstmCellV2Node[i]->GetInDataAnchor(index)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                  innerproductNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                          return FAILED);
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
                innerproductNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);
      }
      index = index + 1;
    }
    if (i == 0 && expose_hidden) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(h0Index)->GetPeerOutAnchor(),
                                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                fusedNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              fusedNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);
      index = index + 1;
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(c0Index)->GetPeerOutAnchor(),
                                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                fusedNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                        return FAILED);
      index = index + 1;
    } else if (i > 0) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmCellV2Node[i - 1]->GetOutDataAnchor(0),
                                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                lstmCellV2Node[i - 1]->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              lstmCellV2Node[i - 1]->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);
      index = index + 1;
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmCellV2Node[i - 1]->GetOutDataAnchor(1),
                                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                lstmCellV2Node[i - 1]->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              lstmCellV2Node[i - 1]->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 0);
      index = index + 1;
    }
    // connect the wxh constOpNode
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(wxIndex)->GetPeerOutAnchor(),
                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 1),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), 0, lstmCellV2Node[i]->GetName().c_str(), 1);
    index = index + 1;
    // concat node
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(lstmCellV2Node[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                lstmCellV2Node[i]->GetName().c_str(), 0, concatNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            concatNode->GetName().c_str(), 0, concatNode->GetName().c_str(), 0);
    // bias
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor(),
                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                lstmCellV2Node[i]->GetName().c_str(), 0, fusedNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), 0, fusedNode->GetName().c_str(), 0);
    index = index + 1;
    // w_xh_deqscale
    if (opType == "AscendQuant") {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(9)->GetPeerOutAnchor(),
                                                           lstmCellV2Node[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                lstmCellV2Node[i]->GetName().c_str(), 0, fusedNode->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              fusedNode->GetName().c_str(), 0, fusedNode->GetName().c_str(), 0);
      index = index + 1;
    }
  }
  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The fanout size of lstmNode is [%d].",
            fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                                concatNode->GetName().c_str(), fusedNode->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              concatNode->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }
  if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The fanout size of lstmNode is [%d].",
            fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmCellV2Node[numSplitX - 1]->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                  lstmCellV2Node[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              lstmCellV2Node[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }
  if (fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The fanout size of lstmNode is [%d].",
            fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmCellV2Node[numSplitX - 1]->GetOutDataAnchor(1), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                  lstmCellV2Node[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              lstmCellV2Node[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }
  // unlink all control input of LSTMD
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of LSTMD
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove LSTMD from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}
REGISTER_PASS("ALSTMFusionPass", BUILT_IN_GRAPH_PASS, ALSTMFusionPass);
}  // namespace fe