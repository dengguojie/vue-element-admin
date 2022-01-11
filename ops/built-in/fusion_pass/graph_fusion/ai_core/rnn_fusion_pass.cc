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
 * \file rnn_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop \brief LayerNormBetaGammaBackprop)
 */
#include "rnn_fusion_pass.h"

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
static const int32_t INT_NUM_TWO = 2;
static const int32_t INT_NUM_PAD = 15;
static const int32_t INT_NUM_CIN = 16;
static const char* FUSED_NODE = "RNN";
static const std::string PATTERN_FUSEDNODE = "RNN";

vector<FusionPattern*> RNNFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("RNNFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::GeTensorDesc RNNFusionPass::ProcessStatic(const ge::NodePtr& fusedNode, const int32_t num_output,
                                              ge::NodePtr& innerproductNode,
                                              ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                              bool& failStatus) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProcessStatic:W_xh_x_static->FullyConnection(InnerProduct).");
  ge::GeTensorDesc outputTensorDesc;
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedDesc is null, fusion failed.");
                    failStatus = true, return outputTensorDesc);
  // inputTensorDesc:x_static
  ge::GeTensorDesc inputTensorDesc = fusedDesc->GetInputDesc(INT_NUM_TWO);
  DataType dataType = inputTensorDesc.GetDataType();

  // add innerproduct desc
  ge::OpDescPtr innerProductStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (innerProductStaticDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() +
                                                             "/W_xh_x_static", "FullyConnection")),
      failStatus = true; return outputTensorDesc);

  // x_static [N,input_size]->[N,input_size,1,1]
  ge::GeShape outputShape = inputTensorDesc.GetShape();
  std::vector<int64_t> dimsInputXShape;
  dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(0));
  dimsInputXShape.push_back(inputTensorDesc.GetShape().GetDim(1));
  dimsInputXShape.push_back(1);
  dimsInputXShape.push_back(1);
  ge::GeShape inputXShape(dimsInputXShape);
  inputTensorDesc.SetShape(inputXShape);
  innerProductStaticDesc->AddInputDesc("x", inputTensorDesc);  // 5HD

  bool expose_hidden = false;
  ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);
  // w_sh[input_size, num_output]
  ge::GeTensorDesc inputWTensorDesc;
  if (expose_hidden) {
    inputWTensorDesc = fusedDesc->GetInputDesc("w_sh");
  } else {
    inputWTensorDesc = fusedDesc->GetInputDesc(5);
  }
  int32_t wshInputSize = inputWTensorDesc.GetShape().GetDim(1);
  int32_t wshNumOutput = inputWTensorDesc.GetShape().GetDim(0);
  int32_t destWshInputSize = (wshInputSize + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  int32_t destWshNumOutput = (wshNumOutput + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  std::vector<int64_t> inputWDims;
  inputWDims.push_back(destWshNumOutput);
  inputWDims.push_back(destWshInputSize);
  inputWDims.push_back(1);
  inputWDims.push_back(1);
  ge::GeShape inputWShape(inputWDims);
  std::vector<int64_t> orgInputWDims;
  orgInputWDims.push_back(wshNumOutput);
  orgInputWDims.push_back(wshInputSize);
  orgInputWDims.push_back(1);
  orgInputWDims.push_back(1);
  ge::GeShape orgInputWShape(orgInputWDims);
  inputWTensorDesc.SetShape(inputWShape);
  inputWTensorDesc.SetOriginShape(orgInputWShape);
  inputWTensorDesc.SetFormat(ge::FORMAT_NCHW);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);

  innerProductStaticDesc->AddInputDesc("w", inputWTensorDesc);  // FRACTAL_Z

  outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_NCHW, dataType);

  // set shape for FRACTAL_NZ(FE auto handle)
  std::vector<int64_t> dimsY;
  dimsY.push_back(inputTensorDesc.GetShape().GetDim(0));
  dimsY.push_back(wshNumOutput);
  dimsY.push_back(1);
  dimsY.push_back(1);
  ge::GeShape dimsYShape(dimsY);
  outputTensorDesc.SetShape(dimsYShape);
  outputTensorDesc.SetOriginShape(dimsYShape);
  outputTensorDesc.SetFormat(ge::FORMAT_NCHW);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  innerProductStaticDesc->AddOutputDesc("y", outputTensorDesc);  // 5HD -> FRACTAL_NZ
  ge::AttrUtils::SetInt(innerProductStaticDesc, "num_output", num_output);
  ge::AttrUtils::SetBool(innerProductStaticDesc, "transpose", false);
  ge::AttrUtils::SetBool(innerProductStaticDesc, "bias_term", false);
  ge::AttrUtils::SetInt(innerProductStaticDesc, "axis", 1);

  // AddNode
  innerproductNode = graph.AddNode(innerProductStaticDesc);
  FUSION_PASS_CHECK(innerproductNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            innerProductStaticDesc->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(innerproductNode);

  return outputTensorDesc;
}

vector<ge::NodePtr> RNNFusionPass::ProcessRnnCell(const ge::NodePtr& fusedNode, ge::ComputeGraph& graph,
                                                  const ge::GeTensorDesc& outInnerProductTensorDesc,
                                                  vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                  const bool has_static) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProcessRnnCell:RNN->BasicRNNCell.");

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  bool expose_hidden = false;
  ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);
  ge::GeTensorDesc biashTensorDesc = fusedDesc->GetInputDesc("bias_h");
  ge::GeTensorDesc biasoTensorDesc = fusedDesc->GetInputDesc("bias_o");
  ge::GeTensorDesc wxhTensorDesc = fusedDesc->GetInputDesc("w_xh");
  ge::GeTensorDesc whohhTensorDesc = fusedDesc->GetInputDesc("w_hh");

  int32_t num_output = 0;
  ge::AttrUtils::GetInt(fusedDesc, "num_output", num_output);

  ge::GeTensorDesc inputTensorDesc0 = fusedDesc->GetInputDesc(0);
  ge::GeShape shape0 = inputTensorDesc0.GetShape();
  int64_t batchDim = shape0.GetDim(1);
  int64_t inputSizeDim = shape0.GetDim(INT_NUM_TWO);
  vector<int64_t> oriShapeDims0;
  oriShapeDims0.push_back(1);
  oriShapeDims0.push_back(batchDim);
  oriShapeDims0.push_back(inputSizeDim);
  GeShape ori_shape_input_0(oriShapeDims0);
  vector<int64_t> shapeDims0;
  shapeDims0.push_back((inputSizeDim + INT_NUM_PAD) / INT_NUM_CIN);
  shapeDims0.push_back((batchDim + INT_NUM_PAD) / INT_NUM_CIN);
  shapeDims0.push_back(INT_NUM_CIN);
  shapeDims0.push_back(INT_NUM_CIN);
  GeShape shape_input_0(shapeDims0);
  ge::GeTensorDesc xInputTensorDesc =
      ge::GeTensorDesc(shape_input_0, ge::FORMAT_FRACTAL_NZ, inputTensorDesc0.GetDataType());
  xInputTensorDesc.SetOriginShape(ori_shape_input_0);
  xInputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorDesc inputContTensorDesc = fusedDesc->GetInputDesc(1);
  ge::GeShape shape1 = inputContTensorDesc.GetShape();
  vector<int64_t> shapeDims1 = {(shape1.GetDim(1) + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN};
  ge::GeShape shapeInputCont(shapeDims1);
  inputContTensorDesc.SetShape(shapeInputCont);
  inputContTensorDesc.SetOriginShape(shapeInputCont);

  DataType dataType = biashTensorDesc.GetDataType();
  int32_t numOutputDim = biashTensorDesc.GetShape().GetDim(0);
  std::vector<int64_t> dimsHShape;
  int32_t hDim1 = (batchDim + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  int32_t hDim2 = (numOutputDim + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  dimsHShape.push_back(hDim1);
  dimsHShape.push_back(hDim2);
  ge::GeShape inputHShape(dimsHShape);
  std::vector<int64_t> dimsHOriginShape;
  dimsHOriginShape.push_back(batchDim);
  dimsHOriginShape.push_back(numOutputDim);
  ge::GeShape inputHOriginShape(dimsHOriginShape);
  ge::GeTensorDesc hTensorDesc = ge::GeTensorDesc(inputHShape, ge::FORMAT_ND, dataType);
  hTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  hTensorDesc.SetOriginShape(inputHOriginShape);
  ge::TensorUtils::SetRealDimCnt(hTensorDesc, INT_NUM_TWO);

  std::vector<int64_t> dimsHShapeH;
  dimsHShapeH.push_back(1);
  dimsHShapeH.push_back(hDim1);
  dimsHShapeH.push_back(hDim2);
  ge::GeShape inputHShapeH(dimsHShapeH);
  std::vector<int64_t> dimsHOriginShapeH;
  dimsHOriginShapeH.push_back(1);
  dimsHOriginShapeH.push_back(batchDim);
  dimsHOriginShapeH.push_back(numOutputDim);
  ge::GeShape inputHOriginShapeH(dimsHOriginShapeH);
  ge::GeTensorDesc hTensorDescH = ge::GeTensorDesc(inputHShapeH, ge::FORMAT_ND, dataType);
  hTensorDescH.SetOriginFormat(ge::FORMAT_ND);
  hTensorDescH.SetOriginShape(inputHOriginShapeH);

  int32_t wxhRow = wxhTensorDesc.GetShape().GetDim(0);
  int32_t wxhCol = wxhTensorDesc.GetShape().GetDim(1);
  int32_t destWxhRow = (wxhRow + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  int32_t destWxhCol = (wxhCol + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  std::vector<int64_t> dimsInputWxhDim;
  dimsInputWxhDim.push_back(destWxhRow);
  dimsInputWxhDim.push_back(destWxhCol);
  ge::GeShape dimsInputWxhShape(dimsInputWxhDim);
  std::vector<int64_t> dimsOriWxhDim;
  dimsOriWxhDim.push_back(wxhRow);
  dimsOriWxhDim.push_back(wxhCol);
  ge::GeShape dimsOriWxhShape(dimsOriWxhDim);
  wxhTensorDesc.SetShape(dimsInputWxhShape);
  wxhTensorDesc.SetOriginShape(dimsOriWxhShape);
  wxhTensorDesc.SetFormat(ge::FORMAT_HWCN);
  wxhTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);

  int32_t whohhRow = whohhTensorDesc.GetShape().GetDim(0);
  int32_t whohhCol = whohhTensorDesc.GetShape().GetDim(1);
  int32_t destWhohhRow = (whohhRow + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  int32_t destWhohhCol = (whohhCol + INT_NUM_PAD) / INT_NUM_CIN * INT_NUM_CIN;
  std::vector<int64_t> dimsInputWhohhDim;
  dimsInputWhohhDim.push_back(destWhohhRow);
  dimsInputWhohhDim.push_back(destWhohhCol);
  ge::GeShape dimsInputWhohhShape(dimsInputWhohhDim);
  std::vector<int64_t> dimsOriWhohhDim;
  dimsOriWhohhDim.push_back(whohhRow);
  dimsOriWhohhDim.push_back(whohhCol);
  ge::GeShape dimsOriWhohhShape(dimsOriWhohhDim);
  whohhTensorDesc.SetShape(dimsInputWhohhShape);
  whohhTensorDesc.SetOriginShape(dimsOriWhohhShape);
  whohhTensorDesc.SetFormat(ge::FORMAT_HWCN);
  whohhTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);

  vector<int64_t> outputOriDims;
  outputOriDims.push_back(1);
  outputOriDims.push_back(batchDim);
  outputOriDims.push_back(numOutputDim);
  GeShape outputOriShape(outputOriDims);

  vector<ge::NodePtr> rnnCellNode = {};

  int64_t tSize = fusedDesc->GetInputDesc(0).GetShape().GetDim(0);
  int64_t tIndex = tSize - 1;
  for (int64_t i = 0; i < tSize; i++) {
    ge::OpDescPtr basicRNNDesc2 = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (basicRNNDesc2 =
             std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/BasicRNNCell" + std::to_string(i), "BasicRNNCell")),
        failStatus = true; return rnnCellNode);

    basicRNNDesc2->AddInputDesc("x", xInputTensorDesc);
    if (expose_hidden || (!expose_hidden && i > 0)) {
      basicRNNDesc2->AddInputDesc("cont", inputContTensorDesc);
    }

    if (has_static) {
      ge::GeShape outInnerProductShape = outInnerProductTensorDesc.GetShape();
      ge::GeTensorDesc staticInputTensorDesc =
          ge::GeTensorDesc(outInnerProductShape, ge::FORMAT_NCHW, outInnerProductTensorDesc.GetDataType());
      std::vector<int64_t> staticDims;
      staticDims.push_back(outInnerProductShape.GetDim(0));
      staticDims.push_back(outInnerProductShape.GetDim(1));
      ge::GeShape staticShape(staticDims);
      staticInputTensorDesc.SetShape(staticShape);
      staticInputTensorDesc.SetOriginShape(staticShape);
      staticInputTensorDesc.SetFormat(ge::FORMAT_NCHW);
      staticInputTensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
      basicRNNDesc2->AddInputDesc("w_xh_x_static", staticInputTensorDesc);
    }

    if (i == 0 && expose_hidden) {
      basicRNNDesc2->AddInputDesc("h_0", hTensorDesc);
    }

    if (i > 0) {
      basicRNNDesc2->AddInputDesc("h_0", hTensorDesc);
      expose_hidden = true;
    }

    basicRNNDesc2->AddInputDesc("w_xh", wxhTensorDesc);
    basicRNNDesc2->AddInputDesc("bias_h", biashTensorDesc);
    if (expose_hidden) {
      basicRNNDesc2->AddInputDesc("w_hh", whohhTensorDesc);
    }
    basicRNNDesc2->AddInputDesc("w_ho", whohhTensorDesc);
    basicRNNDesc2->AddInputDesc("bias_o", biasoTensorDesc);

    ge::AttrUtils::SetInt(basicRNNDesc2, "num_output", num_output);
    ge::AttrUtils::SetBool(basicRNNDesc2, "expose_hidden", expose_hidden);

    if (i == tIndex) {
      ge::GeTensorDesc hTensorDescLast = ge::GeTensorDesc(hTensorDesc.GetShape(), ge::FORMAT_ND, dataType);
      hTensorDescLast.SetOriginShape(outputOriShape);
      hTensorDescLast.SetOriginFormat(ge::FORMAT_ND);
      basicRNNDesc2->AddOutputDesc("o_t", hTensorDescH);
      basicRNNDesc2->AddOutputDesc("h_t", hTensorDescLast);
    } else {
      basicRNNDesc2->AddOutputDesc("o_t", hTensorDescH);
      basicRNNDesc2->AddOutputDesc("h_t", hTensorDesc);
    }

    // add the sub operators to the graph
    ge::NodePtr basicRNNNode = graph.AddNode(basicRNNDesc2);
    FUSION_PASS_CHECK(
        basicRNNNode == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                       basicRNNDesc2->GetName().c_str()),
        failStatus = true);
    newNodes.push_back(basicRNNNode);
    rnnCellNode.push_back(basicRNNNode);
  }
  return rnnCellNode;
}

Status RNNFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "RNNFusionPass Start.");
  // get the NodePtr of RNN
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the OpDescPtr of RNN
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  int32_t num_output = 0;
  ge::AttrUtils::GetInt(fusedDesc, "num_output", num_output);
  ge::NodePtr innerproductNode;
  ge::GeTensorDesc outInnerProductTensorDesc;

  bool expose_hidden = false;
  ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", expose_hidden);

  // handle x_static
  bool has_static = false;
  bool failStatus = false;
  if (fusedDesc->MutableInputDesc(INT_NUM_TWO) != nullptr) {
    has_static = true;
  }

  if (has_static) {
    bool new_expose_hidden = false;
    ge::AttrUtils::GetBool(fusedDesc, "expose_hidden", new_expose_hidden);
    ge::GeTensorDesc inputWTensorDesc;
    if (new_expose_hidden) {
      inputWTensorDesc = fusedDesc->GetInputDesc("w_sh");
    } else {
      inputWTensorDesc = fusedDesc->GetInputDesc(5);
    }
    if (PatternFusionUtil::IsUnknownShape(inputWTensorDesc.GetShape().GetDim(1)) ||
        PatternFusionUtil::IsUnknownShape(inputWTensorDesc.GetShape().GetDim(0))) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RNNFusionPass cannot be applied for unknown shape.");
      return FAILED;
    }

    // x_static->innerproduct
    outInnerProductTensorDesc = ProcessStatic(fusedNode, num_output, innerproductNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(failStatus,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "ProcessStatic:check failed, fusion failed."),
                      return PARAM_INVALID);
  }

  // add splitVd for x
  ge::GeTensorDesc inputTensorDesc0 = fusedDesc->GetInputDesc(0);
  // numSplitX:T
  int32_t numSplitX = inputTensorDesc0.GetShape().GetDim(0);
  // nDim:N
  int32_t nDim = inputTensorDesc0.GetShape().GetDim(1);
  ge::OpDescPtr spiltStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (spiltStaticDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/X_SplitVD", "SplitVD")),
      return PARAM_INVALID);
  ge::GeTensorDesc splitNodeXDesc =
      ge::GeTensorDesc(inputTensorDesc0.GetShape(), inputTensorDesc0.GetFormat(), ge::DT_FLOAT16);
  splitNodeXDesc.SetOriginShape(inputTensorDesc0.GetOriginShape());
  spiltStaticDesc->AddInputDesc("input_value", splitNodeXDesc);
  ge::GeShape shape0 = splitNodeXDesc.GetShape();
  shape0.SetDim(0, 1);
  splitNodeXDesc.SetShape(shape0);
  splitNodeXDesc.SetOriginShape(shape0);
  std::vector<int64_t> sizeSplitsX = {};
  for (int64_t i = 0; i < numSplitX; i++) {
    spiltStaticDesc->AddOutputDesc("output_data" + std::to_string(i + 1), splitNodeXDesc);
    sizeSplitsX.push_back(1);
  }

  ge::AttrUtils::SetListInt(spiltStaticDesc, "size_splits", sizeSplitsX);
  ge::AttrUtils::SetInt(spiltStaticDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(spiltStaticDesc, "num_split", numSplitX);

  // add SplitVD to the graph
  ge::NodePtr splitXNode = graph.AddNode(spiltStaticDesc);
  FUSION_PASS_CHECK(
      splitXNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "fusionNode:%s is null, fusion failed.", spiltStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(splitXNode);

  // add splitVd for cont
  ge::GeTensorDesc inputContTensorDesc = fusedDesc->GetInputDesc(1);
  int32_t numSplitCont = inputContTensorDesc.GetShape().GetDim(0);
  ge::OpDescPtr spiltContStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (spiltContStaticDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/Cont_SplitVD", "SplitVD")),
      return PARAM_INVALID);
  ge::GeTensorDesc splitNodeContDesc =
      ge::GeTensorDesc(inputContTensorDesc.GetShape(), inputContTensorDesc.GetFormat(), ge::DT_FLOAT16);
  splitNodeContDesc.SetOriginShape(inputContTensorDesc.GetOriginShape());
  spiltContStaticDesc->AddInputDesc("input_value", splitNodeContDesc);
  ge::GeShape shape1 = splitNodeContDesc.GetShape();
  shape1.SetDim(0, 1);
  splitNodeContDesc.SetShape(shape1);
  splitNodeContDesc.SetOriginShape(shape1);
  std::vector<int64_t> sizeSplitsCont = {};
  for (int64_t i = 0; i < numSplitCont; i++) {
    spiltContStaticDesc->AddOutputDesc("output_data" + std::to_string(i + 1), splitNodeContDesc);
    sizeSplitsCont.push_back(1);
  }
  ge::AttrUtils::SetListInt(spiltContStaticDesc, "size_splits", sizeSplitsCont);
  ge::AttrUtils::SetInt(spiltContStaticDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(spiltContStaticDesc, "num_split", numSplitCont);

  // add SplitVD to the graph
  ge::NodePtr splitContNode = graph.AddNode(spiltContStaticDesc);
  FUSION_PASS_CHECK(
      splitContNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "fusionNode:%s is null, fusion failed.", spiltContStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(splitContNode);

  // add rnnCellNode
  ge::GeTensorDesc biashTensorDesc = fusedDesc->GetInputDesc("bias_h");
  ge::GeTensorDesc wxhTensorDesc = fusedDesc->GetInputDesc("w_xh");
  ge::GeTensorDesc whohhTensorDesc = fusedDesc->GetInputDesc("w_hh");
  if (PatternFusionUtil::IsUnknownShape(inputTensorDesc0.GetShape().GetDim(1)) ||
      PatternFusionUtil::IsUnknownShape(inputTensorDesc0.GetShape().GetDim(INT_NUM_TWO)) ||
      PatternFusionUtil::IsUnknownShape(inputContTensorDesc.GetShape().GetDim(1)) ||
      PatternFusionUtil::IsUnknownShape(biashTensorDesc.GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(wxhTensorDesc.GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(wxhTensorDesc.GetShape().GetDim(1)) ||
      PatternFusionUtil::IsUnknownShape(whohhTensorDesc.GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(whohhTensorDesc.GetShape().GetDim(1))) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RNNFusionPass cannot be applied for unknown shape.");
    return FAILED;
  }

  vector<ge::NodePtr> rnnCellNode =
      ProcessRnnCell(fusedNode, graph, outInnerProductTensorDesc, newNodes, failStatus, has_static);
  FUSION_PASS_CHECK(failStatus,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "ProcessRnnCell:check failed, fusion failed."),
                    return PARAM_INVALID);

  if (rnnCellNode.empty()) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProcessRnnCell:rnnCellNode is empty, fusion failed.");
    return PARAM_INVALID;
  }
  ge::OpDescPtr rnnCellNode0Desc = rnnCellNode[0]->GetOpDesc();

  // add concat
  ge::OpDescPtr concatStaticDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatStaticDesc = std::make_shared<ge::OpDesc>(fusedDesc->GetName() + "/O_ConcatD", "ConcatD")),
      return PARAM_INVALID);
  for (int64_t i = 0; i < numSplitX; i++) {
    std::vector<int64_t> tempShape;
    tempShape.push_back(1);
    tempShape.push_back(nDim);
    tempShape.push_back(num_output);
    ge::GeShape tempInputShape(tempShape);
    ge::GeTensorDesc temphTensorDesc = ge::GeTensorDesc(tempInputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
    temphTensorDesc.SetShape(tempInputShape);
    temphTensorDesc.SetOriginShape(tempInputShape);
    temphTensorDesc.SetOriginFormat(ge::FORMAT_ND);
    concatStaticDesc->AddInputDesc("input_values" + std::to_string(i + 1), temphTensorDesc);
  }
  std::vector<int64_t> totalDims = {};
  totalDims.push_back(numSplitX);
  totalDims.push_back(nDim);
  totalDims.push_back(num_output);
  ge::GeShape totalShape(totalDims);
  ge::GeTensorDesc totalOutput = ge::GeTensorDesc(totalShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  totalOutput.SetOriginFormat(ge::FORMAT_ND);
  totalOutput.SetShape(totalShape);
  totalOutput.SetOriginShape(totalShape);
  concatStaticDesc->AddOutputDesc("output_data", totalOutput);
  ge::AttrUtils::SetInt(concatStaticDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatStaticDesc, "N", numSplitX);

  // add the sub operators to the graph
  ge::NodePtr concatNode = graph.AddNode(concatStaticDesc);
  FUSION_PASS_CHECK(
      concatNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "fusionNode:%s is null, fusion failed.", concatStaticDesc->GetName().c_str()),
      return PARAM_INVALID);
  newNodes.push_back(concatNode);

  // handle edge
  if (has_static) {
    // edge for innerproduct
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(INT_NUM_TWO)->GetPeerOutAnchor(),
                                           innerproductNode->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0),
        return FAILED);

    int wshIndex = fusedDesc->GetInputIndexByName("w_sh");
    if (!expose_hidden) {
      wshIndex = wshIndex - 1;
    }
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(wshIndex)->GetPeerOutAnchor(),
                                           innerproductNode->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), 0, innerproductNode->GetName().c_str(), 0);
  }

  // edge for spiltVd(x)
  FUSION_PASS_CHECK(
      SUCCESS !=
          ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(), splitXNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              fusedNode->GetName().c_str(), 0, splitXNode->GetName().c_str(), 0),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
          fusedNode->GetName().c_str(), 0, splitXNode->GetName().c_str(), 0);

  // edge for spiltVd(cont)
  FUSION_PASS_CHECK(
      SUCCESS !=
          ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(), splitContNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              fusedNode->GetName().c_str(), 1, splitContNode->GetName().c_str(), 0),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
          fusedNode->GetName().c_str(), 1, splitContNode->GetName().c_str(), 0);

  // input indexs
  int wxhIndex = 4;
  int biasHIndex = 5;
  int whhIndex = 7;
  int whoIndex = 8;
  int biasOIndex = 9;

  // edge for rnncell
  for (int64_t i = 0; i < numSplitX; i++) {
    int index = 0;
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(splitXNode->GetOutDataAnchor(i), rnnCellNode[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                splitXNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            splitXNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0);
    index = index + 1;

    if (expose_hidden || (!expose_hidden && i > 0)) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitContNode->GetOutDataAnchor(i),
                                                           rnnCellNode[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%lu] to fusion node:%s's input[%d] failed.",
                                splitContNode->GetName().c_str(), i, rnnCellNode[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              splitContNode->GetName().c_str(), i, rnnCellNode[i]->GetName().c_str(), 0);
      index = index + 1;
    }

    if (has_static) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(innerproductNode->GetOutDataAnchor(0),
                                                           rnnCellNode[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                innerproductNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              innerproductNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0);

      index = index + 1;
    }

    if (i == 0 && expose_hidden) {
      // handle h_0 for first time
      int hIndex = 3;
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(hIndex)->GetPeerOutAnchor(),
                                                           rnnCellNode[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                fusedNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              fusedNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0);
      index = index + 1;
    } else if (i > 0) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(rnnCellNode[i - 1]->GetOutDataAnchor(1),
                                                           rnnCellNode[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                rnnCellNode[i - 1]->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              rnnCellNode[i - 1]->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 0);
      index = index + 1;
    }

    // edge for w_xh
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(wxhIndex)->GetPeerOutAnchor(),
                                           rnnCellNode[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 1),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), wxhIndex, rnnCellNode[i]->GetName().c_str(), index);
    index = index + 1;

    // edge for bias_h
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(biasHIndex)->GetPeerOutAnchor(),
                                           rnnCellNode[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                rnnCellNode[i]->GetName().c_str(), biasHIndex, fusedNode->GetName().c_str(), index),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), biasHIndex, fusedNode->GetName().c_str(), index);
    index = index + 1;

    if (expose_hidden || i > 0) {
      // edge for w_hh
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(whhIndex)->GetPeerOutAnchor(),
                                                           rnnCellNode[i]->GetInDataAnchor(index)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                fusedNode->GetName().c_str(), whhIndex, rnnCellNode[i]->GetName().c_str(), index),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
              fusedNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 1);
      index = index + 1;
    }

    // edge for w_ho
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(whoIndex)->GetPeerOutAnchor(),
                                           rnnCellNode[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), 0, rnnCellNode[i]->GetName().c_str(), 1),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), whoIndex, rnnCellNode[i]->GetName().c_str(), index);
    index = index + 1;

    // edge for bias_o
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(biasOIndex)->GetPeerOutAnchor(),
                                           rnnCellNode[i]->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                rnnCellNode[i]->GetName().c_str(), biasOIndex, fusedNode->GetName().c_str(), index),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), biasOIndex, fusedNode->GetName().c_str(), index);

    // edge for concat
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(rnnCellNode[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                rnnCellNode[i]->GetName().c_str(), 0, concatNode->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            concatNode->GetName().c_str(), 0, concatNode->GetName().c_str(), 0);
  }

  // edge for o
  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      if (inAnchorPtr != nullptr) {
        inAnchorPtr->UnlinkAll();
      }
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                                concatNode->GetName().c_str(), fusedNode->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              concatNode->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }

  // edge for h
  if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      if (inAnchorPtr != nullptr) {
        inAnchorPtr->UnlinkAll();
      }
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(rnnCellNode[numSplitX - 1]->GetOutDataAnchor(1), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                  rnnCellNode[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              rnnCellNode[numSplitX - 1]->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }

  // release ControlAnchor InDataAnchors
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove rnn
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "RNNFusionPass End.");
  return SUCCESS;
}

REGISTER_PASS("RNNFusionPass", BUILT_IN_GRAPH_PASS, RNNFusionPass);
}  // namespace fe
