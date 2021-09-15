/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file avg_pool_v2_pass.cpp
 * \brief avgPool_v2 fusion pass
 */
#include "avg_pool_v2_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "error_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const uint16_t UINT_NUM_ONE = 1;
static const uint16_t UINT_NUM_ZERO = 0;
static const string PATTERN_AVGPOOL = "AvgPoolV2";
static const std::string CONSTANTOP = "Const";
static const char* AVGPOOL = "AvgPoolV2";
static const int64_t COUT = 16;
static const int64_t CIN = 16;
// kernel_h*kernel_w
static const int64_t AVGV2_KERNEL_SIZE_H_MUL_W = 255;
// ksize restrictions
static const int64_t AVGV2_KERNEL_SIZE = 20;

vector<FusionPattern*> AvgPoolV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define AvgPoolFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AvgPoolV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_AVGPOOL, {AVGPOOL}).SetOutput(PATTERN_AVGPOOL);

  patterns.push_back(pattern);

  return patterns;
}

NodePtr AvgPoolV2FusionPass::AddMul(ge::ComputeGraph& graph, ge::NodePtr& avgPoolNode, ge::Format& inputOriginFormat) {
  ge::OutDataAnchorPtr avgPoolAnchorPtr1 = avgPoolNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(avgPoolAnchorPtr1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolAnchorPtr1 is null, fusion failed."),
                    return nullptr);
  ge::NodePtr postNode = nullptr;
  ge::NodePtr mulNode = nullptr;
  int64_t mulN = 0;
  int64_t mulH = 0;
  int64_t mulW = 0;
  int64_t mulC = 0;
  int64_t mulC1 = 0;

  // creat a mul node
  std::shared_ptr<ge::OpDesc> mulDesc = nullptr;
  mulDesc = std::make_shared<ge::OpDesc>(avgPoolNode->GetName() + "_mul_layer", "Mul");
  FUSION_PASS_CHECK(mulDesc == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulDesc is null, mul failed."), return nullptr);

  // add input
  ge::GeTensorDesc input_desc = avgPoolNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape mulShape = input_desc.GetShape();
  vector<int64_t> dimMul = mulShape.GetDims();

  if (dimMul.size() == 4) {
    if (inputOriginFormat == FORMAT_NHWC) {
      mulN = dimMul[0];
      mulH = dimMul[1];
      mulW = dimMul[2];
      mulC = dimMul[3];
    } else {
      mulN = dimMul[0];
      mulH = dimMul[2];
      mulW = dimMul[3];
      mulC = dimMul[1];
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimMul is invalid, please check!");
    return nullptr;
  }

  mulC1 = (mulC + 16 - 1) / 16;
  vector<int64_t> mulDimInfo = {mulN, mulC1, mulH, mulW, 16};

  ge::GeShape mulInputShape(mulDimInfo);
  input_desc.SetShape(mulInputShape);
  input_desc.SetOriginShape(mulShape);
  input_desc.SetFormat(ge::FORMAT_NC1HWC0);
  input_desc.SetOriginFormat(inputOriginFormat);
  FUSION_PASS_CHECK(mulDesc->AddInputDesc(input_desc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc input failed."), return nullptr);

  // add output
  FUSION_PASS_CHECK(mulDesc->AddOutputDesc(input_desc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc output failed."), return nullptr);

  // add node
  mulNode = graph.AddNode(mulDesc);

  for (auto postAnchorPtr0 : avgPoolAnchorPtr1->GetPeerInDataAnchors()) {
    postNode = postAnchorPtr0->GetOwnerNode();

    // remove edge between avgpool and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, avgPoolAnchorPtr1) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                            mulNode->GetName().c_str(), postNode->GetName().c_str()),
                      return nullptr);
  }
  // add edge between avgpool and mul
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(avgPoolAnchorPtr1, mulNode->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                          avgPoolNode->GetName().c_str(), mulNode->GetName().c_str()),
                    return nullptr);

  return mulNode;
}

Status AvgPoolV2FusionPass::GenCoffeFP16(const vector<int64_t> shape, vector<int64_t> window, vector<int64_t> stride,
                                         vector<int64_t> pad, const int64_t dimH, const int64_t dimW,
                                         uint16_t& output1) {
  uint16_t* output = &output1;
  int64_t h_start = 0;
  int64_t w_start = 0;
  int64_t h_end = 0;
  int64_t w_end = 0;
  float area = 0;

  for (int m = 0; m < shape[0]; m++) {
    for (int n = 0; n < shape[1]; n++) {
      for (int64_t i = 0; i < shape[2]; i++) {
        for (int64_t j = 0; j < shape[3]; j++) {
          for (int k = 0; k < shape[4]; k++) {
            h_start = i * stride[0] - pad[0];
            w_start = j * stride[1] - pad[2];
            h_end = min(h_start + window[0], dimH);
            w_end = min(w_start + window[1], dimW);
            h_start = max(h_start, static_cast<int64_t>(0));
            w_start = max(w_start, static_cast<int64_t>(0));
            area = max((h_end - h_start) * (w_end - w_start), static_cast<int64_t>(1));
            area = 1.0 / area;
            fp16_t a;
            a.val = area;
            fp16_t b;
            b.val = 0;
            b = (float)area;
            output[m * (shape[1] * shape[2] * shape[3] * shape[4]) + n * (shape[2] * shape[3] * shape[4]) +
                   i * (shape[3] * shape[4]) + j * shape[4] + k] = b.val;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status AvgPoolV2FusionPass::AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mulNode, string& padding,
                                     vector<int64_t>& dimInfo, vector<int64_t> ksize, vector<int64_t> stride,
                                     vector<int64_t> pads) {
  int64_t outputN = 0;
  int64_t outputH = 0;
  int64_t outputW = 0;
  int64_t outputC = 0;
  int64_t outputC1 = 0;
  int64_t outputC0 = 16;
  int64_t dimH = 0;
  int64_t dimW = 0;
  int64_t padRow = 0;
  int64_t padCol = 0;
  int64_t padTop = 0;
  int64_t padBottom = 0;
  int64_t padLeft = 0;
  int64_t padRight = 0;
  vector<int64_t> pad;
  vector<int64_t> dilation;
  pad = pads;
  dilation = {1, 1};

  ge::OpDescPtr mulOp = mulNode->GetOpDesc();
  FUSION_PASS_CHECK(mulOp == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulOp is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc inputDesc0 = mulOp->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  ge::GeShape outputShape = inputDesc0.GetOriginShape();
  vector<int64_t> dimOut = outputShape.GetDims();
  if (dimOut.size() == 4) {
    if (inputDesc0OriginFormat == FORMAT_NHWC) {
      outputN = dimOut[0];
      outputH = dimOut[1];
      outputW = dimOut[2];
      outputC = dimOut[3];
      dimH = dimInfo[1];
      dimW = dimInfo[2];
    } else {
      outputN = dimOut[0];
      outputH = dimOut[2];
      outputW = dimOut[3];
      outputC = dimOut[1];
      dimH = dimInfo[2];
      dimW = dimInfo[3];
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "dimOut is invalid, please check! actual is %zu, expect is 4", dimOut.size());
    return PARAM_INVALID;
  }
  outputC1 = (outputC + outputC0 - 1) / outputC0;
  if (padding == "SAME") {
    padRow = (outputH - 1) * stride[0] + ((ksize[0] - 1) * dilation[0] + 1) - dimH;
    padCol = (outputW - 1) * stride[1] + ((ksize[1] - 1) * dilation[1] + 1) - dimW;
    padTop = padRow / 2;
    padBottom = padRow - padTop;
    padLeft = padCol / 2;
    padRight = padCol - padLeft;
    padTop = max(padTop, static_cast<int64_t>(0));
    padBottom = max(padBottom, static_cast<int64_t>(0));
    padLeft = max(padLeft, static_cast<int64_t>(0));
    padRight = max(padRight, static_cast<int64_t>(0));
    pad = {padTop, padBottom, padLeft, padRight};
  }

  ge::GeTensorPtr coffePtr = nullptr;
  int64_t coffeSize = outputN * outputC1 * outputH * outputW * outputC0;
  FUSION_PASS_CHECK(coffeSize <= 0,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "coffeSize is Invalid"), return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[coffeSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret = NnSet(coffeSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

  vector<int64_t> coffeDimInfo = {outputN, outputC1, outputH, outputW, outputC0};
  ret = GenCoffeFP16(coffeDimInfo, ksize, stride, pad, dimH, dimW, *inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);

  vector<int64_t> coffeDimInfoOrigin;
  if (inputDesc0OriginFormat == FORMAT_NHWC) {
    coffeDimInfoOrigin = {outputN, outputH, outputW, outputC};
  } else if (inputDesc0OriginFormat == FORMAT_NCHW) {
    coffeDimInfoOrigin = {outputN, outputC, outputH, outputW};
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
      "format is wrong, please check!expect is NHWC or NCHW, actual is %d", inputDesc0OriginFormat);
    return PARAM_INVALID;
  }

  // set const node shape
  ge::GeTensorDesc coffeDesc;
  ge::GeShape coffeShape(coffeDimInfo);
  ge::GeShape coffeShapeOrigin(coffeDimInfoOrigin);
  coffeDesc.SetShape(coffeShape);
  coffeDesc.SetDataType(ge::DT_FLOAT16);
  coffeDesc.SetFormat(ge::FORMAT_NC1HWC0);
  coffeDesc.SetOriginFormat(inputDesc0OriginFormat);
  coffeDesc.SetOriginShape(coffeShapeOrigin);
  coffeDesc.SetOriginDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED((coffePtr = std::make_shared<ge::GeTensor>(
                               coffeDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), coffeSize * sizeof(uint16_t))),
                          coffePtr = nullptr;
                          return PARAM_INVALID);
  ge::OpDescPtr mulDesc = mulNode->GetOpDesc();
  FUSION_PASS_CHECK(mulDesc == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {coffePtr};
  ge::OpDescUtils::SetWeights(mulNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(mulNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
    return PARAM_INVALID;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
}

bool CheckGlobal(const string& padding, int64_t inputH, int64_t inputW, int64_t ksizeH, int64_t ksizeW,
                 int64_t stridesH, int64_t stridesW, vector<int64_t>& pads, bool global_pooling) {
  if (global_pooling) {
    return true;
  }

  if ((inputH == ksizeH) && (inputW == ksizeW)) {
    for (size_t i = 0; i < pads.size(); i++) {
      if (pads[i] != 0) {
        return false;
      }
    }
    if (padding != "SAME"){
      return true;
    } else if (inputH == stridesH && inputW == stridesW){
      return true;
    }
  }
  return false;
}

Status GenerateFilterFP16V2(const int64_t size, const float areaFactor, uint16_t& output1) {
  uint16_t* output = &output1;
  fp16_t t;
  t.val = areaFactor;
  fp16_t tmp2;
  tmp2.val = 0;
  tmp2 = (float)areaFactor;
  for (int i = 0; i < size; i++) {
    output[i] = tmp2.val;
  }
  return SUCCESS;
}

Status GenerateFilterFP16DynamicV2(const vector<int64_t>& shape, const float areaFactor, uint16_t& output1) {
  uint16_t* output = &output1;
  fp16_t area_factor;
  area_factor.val = 0;
  area_factor = static_cast<float>(areaFactor);
  for (int64_t i = 0; i < shape[0]; i++) {
    for (int64_t j = 0; j < shape[1]; j++) {
      for (int64_t k = 0; k < shape[2]; k++) {
        for (int64_t l = 0; l < shape[3]; l++) {
          if (k == l) {
            output[i * (shape[1] * shape[2] * shape[3]) + j * (shape[2] * shape[3]) + k * shape[3] + l] =
                                                                                                      area_factor.val;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status AvgPoolV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // avgpool node
  ge::NodePtr avgPoolNode = GetNodeFromMapping(PATTERN_AVGPOOL, mapping);
  FUSION_PASS_CHECK(avgPoolNode == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolV2Node is null, fusion failed."),
                    return PARAM_INVALID);

  // input of AvgPool
  ge::OpDescPtr avgPoolDesc = avgPoolNode->GetOpDesc();
  FUSION_PASS_CHECK(avgPoolDesc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolV2Node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc avgPoolInputTensor = avgPoolNode->GetOpDesc()->GetInputDesc(0);

  // get shape
  ge::GeShape avgPoolInputShape = avgPoolInputTensor.GetShape();
  ge::Format inputOriginFormat = avgPoolInputTensor.GetOriginFormat();
  // geshape->vector
  vector<int64_t> dimInfo = avgPoolInputShape.GetDims();
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  bool isDynamic = false;
  // when static op or dynamic op phase_running, is_dynamic = false
  if (find(dimInfo.begin(),dimInfo.end(), -1) != dimInfo.end()) {
    isDynamic = true;
  }
  if (dimInfo.size() == 4) {
    if (inputOriginFormat == FORMAT_NHWC) {
      inputC = dimInfo[3];
      inputH = dimInfo[1];
      inputW = dimInfo[2];
    } else if (inputOriginFormat == FORMAT_NCHW) {
      inputC = dimInfo[1];
      inputH = dimInfo[2];
      inputW = dimInfo[3];
    }
    if (PatternFusionUtil::IsUnknownShape(inputC)) {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimInfo is invalid, please check!");
    return PARAM_INVALID;
  }

  string dataFormat;
  string padding;
  vector<int64_t> ksize;
  vector<int64_t> strides;
  vector<int64_t> window;
  vector<int64_t> stride;
  int64_t ksizeH = 0;
  int64_t ksizeW = 0;
  int64_t stridesH = 0;
  int64_t stridesW = 0;
  bool global_pooling;
  bool ceil_mode;
  bool exclusive;
  vector<int64_t> pads;
  // get windowsize padding strides value dataFormat
  ge::AttrUtils::GetStr(avgPoolDesc, "data_format", dataFormat);
  ge::AttrUtils::GetStr(avgPoolDesc, "padding_mode", padding);
  ge::AttrUtils::GetListInt(avgPoolDesc, "ksize", ksize);
  ge::AttrUtils::GetListInt(avgPoolDesc, "strides", strides);
  ge::AttrUtils::GetBool(avgPoolDesc, "global_pooling", global_pooling);
  ge::AttrUtils::GetBool(avgPoolDesc, "ceil_mode", ceil_mode);
  ge::AttrUtils::GetBool(avgPoolDesc, "exclusive", exclusive);
  ge::AttrUtils::GetListInt(avgPoolDesc, "pads", pads);

  if (dataFormat == "NHWC") {
    if (ksize.size() == 4 and strides.size() == 4) {
      ksizeH = ksize[1];
      ksizeW = ksize[2];
      stridesH = strides[1];
      stridesW = strides[2];
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ksize or strides is invalid, please check!");
      return PARAM_INVALID;
    }
  } else if (dataFormat == "NCHW") {
    if (ksize.size() == 4 and strides.size() == 4) {
      ksizeH = ksize[2];
      ksizeW = ksize[3];
      stridesH = strides[2];
      stridesW = strides[3];
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ksize or strides is invalid, please check!");
      return PARAM_INVALID;
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dataFormat is invalid, please check!");
    return PARAM_INVALID;
  }

  window = {ksizeH, ksizeW};
  stride = {stridesH, stridesW};
  // judge global pooling
  bool is_global = false;
  is_global = CheckGlobal(padding, inputH, inputW, ksizeH, ksizeW, stridesH, stridesW, pads, global_pooling);
  if (!isDynamic && is_global) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "avg_pool_v2 is global, graph not changed.");
    return NOT_CHANGED;
  }
  if(stridesH > 63 || stridesW > 63){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "strided_h or strided_w >63, not support");
    return NOT_CHANGED;
  }
  bool AicoreSupport = true;
  AicoreSupport = CheckOpSupported(avgPoolDesc);
  if (!isDynamic && !AicoreSupport) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "aicore not support");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(avgPoolNode->GetOpDesc(), "groups", inputC),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "set groups attr failed"),
                    return FAILED);
  // get pre node of pooling
  ge::InDataAnchorPtr poolingAnchorPtr0 = avgPoolNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr preAnchorPtr0 = poolingAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();

  int64_t matrixSize = inputC * 1 * ksizeH * ksizeW;
  int64_t inputC1 = (inputC + COUT - 1) / COUT;
  if (isDynamic) {
    matrixSize = inputC1 * ksizeH * ksizeW * CIN * COUT;
  }
  FUSION_PASS_CHECK(matrixSize <= 0,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"), return PARAM_INVALID);

  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[matrixSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret;
  ret = NnSet(matrixSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
  vector<int64_t> assitDimInfoOrigin = {inputC, 1, ksizeH, ksizeW};
  vector<int64_t> assitDimInfoDynamic = {inputC1 * ksizeH * ksizeW, 1, CIN, COUT};
  vector<int64_t> assitDimInfoOriginDynamic = {inputC, 1, ksizeH, ksizeW};
  if (!exclusive || padding == "VALID") {
    float areaFactor = 1.0 / (ksizeH * ksizeW);
    // generate one matrix
    if (!isDynamic) {
      ret = GenerateFilterFP16V2(matrixSize, areaFactor, *inputAssit.get());
    } else {
      areaFactor = 1.0;
      ret = GenerateFilterFP16DynamicV2(assitDimInfoDynamic, areaFactor, *inputAssit.get());
    }
    FUSION_PASS_CHECK(ret != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."), return ret);
  } else if (padding == "SAME") {
    float areaFactor = 1.0;
    // generate one matrix
    if (!isDynamic) {
      ret = GenerateFilterFP16V2(matrixSize, areaFactor, *inputAssit.get());
    } else {
      ret = GenerateFilterFP16DynamicV2(assitDimInfoDynamic, areaFactor, *inputAssit.get());
    }
    FUSION_PASS_CHECK(ret != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."), return ret);
    if (!isDynamic) {
      // judge for unknown shape
      int64_t mulC = 0;
      vector<int64_t> dimMul = avgPoolNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
      if (dimMul.size() == 4) {
        if (inputOriginFormat == FORMAT_NHWC) {
          mulC = dimMul[3];
        } else {
          mulC = dimMul[1];
        }
        if (PatternFusionUtil::IsUnknownShape(mulC)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      ge::NodePtr mulNode = AddMul(graph, avgPoolNode, inputOriginFormat);
      FUSION_PASS_CHECK(mulNode == nullptr,
                        CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      // judge input dims for unknown shape
      ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      ge::GeShape outputShape = inputDesc0.GetOriginShape();
      vector<int64_t> dimOut = outputShape.GetDims();
      for (size_t i = 0; i <= 3; i++) {
        auto dim = dimOut[i];
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      FUSION_PASS_CHECK(AddCoffe(graph, mulNode, padding, dimInfo, window, stride) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
    }
  } else if (padding == "CALCULATED") {
    float areaFactor = 1.0;
    // generate one matrix
    if (!isDynamic) {
      ret = GenerateFilterFP16V2(matrixSize, areaFactor, *inputAssit.get());
    } else {
      ret = GenerateFilterFP16DynamicV2(assitDimInfoDynamic, areaFactor, *inputAssit.get());
    }
    FUSION_PASS_CHECK(ret != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."), return ret);
    if (!isDynamic) {
      // judge for unknown shape
      int64_t mulC = 0;
      vector<int64_t> dimMul = avgPoolNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
      if (dimMul.size() == 4) {
        if (inputOriginFormat == FORMAT_NHWC) {
          mulC = dimMul[3];
        } else {
          mulC = dimMul[1];
        }
        if (PatternFusionUtil::IsUnknownShape(mulC)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      ge::NodePtr mulNode = AddMul(graph, avgPoolNode, inputOriginFormat);
      FUSION_PASS_CHECK(mulNode == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      // judge input dims for unknown shape
      ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      ge::GeShape outputShape = inputDesc0.GetOriginShape();
      vector<int64_t> dimOut = outputShape.GetDims();
      for (size_t i = 0; i <= 3; i++) {
        auto dim = dimOut[i];
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      FUSION_PASS_CHECK(AddCoffe(graph, mulNode, padding, dimInfo, window, stride, pads) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "padding is wrong, please check!");
    return PARAM_INVALID;
  }
  FUSION_PASS_CHECK(ret != SUCCESS,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."), return ret);
  GeTensorDesc tensorDesc;
  ge::GeShape assitShape(assitDimInfoOrigin);
  ge::GeShape assitShapeOrigin(assitDimInfoOrigin);
  ge::GeShape assitShapeDynamic(assitDimInfoDynamic);
  ge::GeShape assitShapeOriginDynamic(assitDimInfoOriginDynamic);
  if (!isDynamic) {
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOrigin);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
  } else {
    tensorDesc.SetShape(assitShapeDynamic);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetFormat(ge::FORMAT_FRACTAL_Z);
    tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOriginDynamic);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
  }

  ge::GeTensorPtr assitPtr = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 matrixSize * sizeof(uint16_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(avgPoolNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(avgPoolNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
    return PARAM_INVALID;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
}

REGISTER_PASS("AvgPoolV2FusionPass", BUILT_IN_GRAPH_PASS, AvgPoolV2FusionPass);
}  // namespace fe
