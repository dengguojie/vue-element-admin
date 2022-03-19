/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include <map>
#include <string>
#include <vector>

#include "error_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "quant_host_cpu_op_common.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const uint16_t UINT_NUM_ONE = 1;
static const uint16_t UINT_NUM_ZERO = 0;
static const int8_t INT8_NUM_ONE = 1;
static const string PATTERN_AVGPOOL = "AvgPoolV2";
static const string CONSTANTOP = "Const";
static const char *AVGPOOL = "AvgPoolV2";
static const int64_t COUT = 16;
static const int64_t CIN = 16;
const int32_t INDEX_CO_avg = 1;
const int32_t INDEX_CI_avg = 0;
// kernel_h*kernel_w
static const int64_t AVGV2_KERNEL_SIZE_H_MUL_W = 255;
// ksize restrictions
static const int64_t AVGV2_KERNEL_SIZE = 20;

vector<FusionPattern *> AvgPoolV2FusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  // define AvgPoolFusion
  FusionPattern *pattern = new (nothrow) FusionPattern("AvgPoolV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_AVGPOOL, {AVGPOOL}).SetOutput(PATTERN_AVGPOOL);

  patterns.push_back(pattern);

  return patterns;
}

NodePtr AvgPoolV2FusionPass::AddMul(ComputeGraph &graph, const NodePtr &avgPoolNode, Format &inputOriginFormat) {
  OutDataAnchorPtr avgPoolAnchorPtr1 = avgPoolNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(avgPoolAnchorPtr1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolAnchorPtr1 is null, fusion failed."),
                    return nullptr);
  NodePtr postNode = nullptr;
  NodePtr mulNode = nullptr;
  int64_t mulN = 0;
  int64_t mulH = 0;
  int64_t mulW = 0;
  int64_t mulC = 0;
  int64_t mulC1 = 0;

  // creat a mul node
  shared_ptr<ge::OpDesc> mulDesc = nullptr;
  mulDesc = make_shared<ge::OpDesc>(avgPoolNode->GetName() + "_mul_layer", "Mul");
  FUSION_PASS_CHECK(mulDesc == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulDesc is null, mul failed."),
                    return nullptr);

  // add input
  GeTensorDesc input_desc = avgPoolNode->GetOpDesc()->GetOutputDesc(0);
  GeShape mulShape = input_desc.GetShape();
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

  GeShape mulInputShape(mulDimInfo);
  input_desc.SetShape(mulInputShape);
  input_desc.SetOriginShape(mulShape);
  input_desc.SetFormat(FORMAT_NC1HWC0);
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
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(postAnchorPtr0, avgPoolAnchorPtr1) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                            mulNode->GetName().c_str(), postNode->GetName().c_str()),
                      return nullptr);
  }
  // add edge between avgpool and mul
  FUSION_PASS_CHECK(GraphUtils::AddEdge(avgPoolAnchorPtr1, mulNode->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                          avgPoolNode->GetName().c_str(), mulNode->GetName().c_str()),
                    return nullptr);

  return mulNode;
}

Status AvgPoolV2FusionPass::GenCoffeFP16(const vector<int64_t> shape, vector<int64_t> window, vector<int64_t> stride,
                                         vector<int64_t> pad, const int64_t dimH, const int64_t dimW,
                                         uint16_t &output1) {
  uint16_t *output = &output1;
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

Status AvgPoolV2FusionPass::AddCoffe(NodePtr &mulNode, const string &padding, vector<int64_t> &dimInfo,
                                     vector<int64_t> ksize, vector<int64_t> stride, vector<int64_t> pads) {
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

  OpDescPtr mulOp = mulNode->GetOpDesc();
  FUSION_PASS_CHECK(mulOp == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulOp is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc inputDesc0 = mulOp->GetInputDesc(0);
  Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  GeShape outputShape = inputDesc0.GetOriginShape();
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
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimOut is invalid, please check! actual is %zu, expect is 4",
                          dimOut.size());
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

  GeTensorPtr coffePtr = nullptr;
  int64_t coffeSize = outputN * outputC1 * outputH * outputW * outputC0;
  FUSION_PASS_CHECK(coffeSize <= 0, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "coffeSize is Invalid"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssit(new (nothrow) uint16_t[coffeSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret = NnSet(coffeSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

  vector<int64_t> coffeDimInfo = {outputN, outputC1, outputH, outputW, outputC0};
  ret = GenCoffeFP16(coffeDimInfo, ksize, stride, pad, dimH, dimW, *inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);

  vector<int64_t> coffeDimInfoOrigin;
  if (inputDesc0OriginFormat == FORMAT_NHWC) {
    coffeDimInfoOrigin = {outputN, outputH, outputW, outputC};
  } else if (inputDesc0OriginFormat == FORMAT_NCHW) {
    coffeDimInfoOrigin = {outputN, outputC, outputH, outputW};
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "format is wrong, please check!expect is NHWC or NCHW, actual is %d",
                          inputDesc0OriginFormat);
    return PARAM_INVALID;
  }

  // set const node shape
  GeTensorDesc coffeDesc;
  GeShape coffeShape(coffeDimInfo);
  GeShape coffeShapeOrigin(coffeDimInfoOrigin);
  coffeDesc.SetShape(coffeShape);
  coffeDesc.SetDataType(DT_FLOAT16);
  coffeDesc.SetFormat(FORMAT_NC1HWC0);
  coffeDesc.SetOriginFormat(inputDesc0OriginFormat);
  coffeDesc.SetOriginShape(coffeShapeOrigin);
  coffeDesc.SetOriginDataType(DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED((coffePtr = make_shared<GeTensor>(coffeDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                            coffeSize * sizeof(uint16_t))),
                          coffePtr = nullptr;
                          return PARAM_INVALID);
  OpDescPtr mulDesc = mulNode->GetOpDesc();
  FUSION_PASS_CHECK(mulDesc == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  vector<GeTensorPtr> weights = {coffePtr};
  OpDescUtils::SetWeights(mulNode, weights);
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

bool CheckGlobal(const string &padding, int64_t inputH, int64_t inputW, int64_t ksizeH, int64_t ksizeW,
                 int64_t stridesH, int64_t stridesW, vector<int64_t> &pads, bool global_pooling) {
  if (global_pooling) {
    return true;
  }

  if ((inputH == ksizeH) && (inputW == ksizeW)) {
    for (size_t i = 0; i < pads.size(); i++) {
      if (pads[i] != 0) {
        return false;
      }
    }
    if (padding != "SAME") {
      return true;
    } else if (inputH == stridesH && inputW == stridesW) {
      return true;
    }
  }
  return false;
}

Status GenerateFilterFP16V2(const int64_t size, const float areaFactor, uint16_t &output1) {
  uint16_t *output = &output1;
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

Status GenerateFilterFP16DynamicV2(const vector<int64_t> &shape, const float areaFactor, uint16_t &output1) {
  uint16_t *output = &output1;
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

Status AvgPoolV2FusionPass::Calc4DWeightAvgPool(const vector<int64_t> &filterDims4D, const int64_t &kernelDataCount,
                                                const int8_t *filterInt8Data, unique_ptr<int32_t[]> &weightInt8Temp) {
  FUSION_PASS_CHECK(filterDims4D.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterDims4D is empty!"), return FAILED);

  if (INDEX_CO_avg > filterDims4D.size()) {
    return FAILED;
  }
  for (int64_t j = 0; j < filterDims4D[INDEX_CO_avg]; j++) {
    int64_t sum_temp = 0;
    for (int64_t i = 0; i < filterDims4D[INDEX_CI_avg]; i++) {
      for (int64_t h = 0; h < filterDims4D[INDEX_FILTER_H]; h++) {
        for (int64_t w = 0; w < filterDims4D[INDEX_FILTER_W]; w++) {
          int64_t k = (j * filterDims4D[INDEX_CI_avg] * filterDims4D[INDEX_FILTER_H] * filterDims4D[INDEX_FILTER_W]) +
                      (i * filterDims4D[INDEX_FILTER_H] * filterDims4D[INDEX_FILTER_W]) +
                      (h * filterDims4D[INDEX_FILTER_W]) + w;
          FUSION_PASS_CHECK(k >= kernelDataCount,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                           "The index %ld is out of weightInt8Data's range", k),
                            return FAILED);
          sum_temp += filterInt8Data[k];
        }
      }
    }
    weightInt8Temp[j] = sum_temp;
  }
  return SUCCESS;
}

Status AvgPoolV2FusionPass::GetWeightOfConvAvgpool(const string &opName, const int8_t *filterInt8Data,
                                                   const vector<int64_t> &filterDims,
                                                   unique_ptr<int32_t[]> &weightInt8OutParam) {
  // get weight_int8
  const int64_t min = 0;
  const int64_t div = 4;
  vector<int64_t> filterDims4D;
  size_t sizeOfFilter = filterDims.size();
  for (uint32_t i = 0; i <= INDEX_FILTER_W; i++) {
    if (i < sizeOfFilter) {
      filterDims4D.emplace_back(filterDims[i]);
    } else {
      filterDims4D.emplace_back(0);
    }
  }

  if (!filterDims4D.empty() && filterDims4D.size() >= INDEX_FILTER_W) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "quant bias optimize, weight shape is NCHW[%ld %ld %ld %ld].",
            filterDims4D[INDEX_CI_avg], filterDims4D[INDEX_CO_avg], filterDims4D[INDEX_FILTER_H],
            filterDims4D[INDEX_FILTER_W]);
  }

  // get conv core kerneldata count
  int64_t kernelDataCount = 1;
  FUSION_PASS_CHECK(GetkernelDataCountForPass(filterDims, kernelDataCount) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetkernelDataCount faild."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(filterInt8Data == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightInt8Data is nullptr"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount <= min,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "kernelDataCount is not a positive number."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount == min || kernelDataCount >= UINT_MAX / div,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "kernelDataCount is out of range."),
                    return PARAM_INVALID);

  // calc weight: accumulate weights
  unique_ptr<int32_t[]> weightInt8Temp(new (nothrow) int32_t[filterDims4D[INDEX_CO_avg]]());
  FUSION_PASS_CHECK(weightInt8Temp == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightInt8Temp is nullptr"),
                    return PARAM_INVALID);
  Status ret;
  FUSION_PASS_CHECK(filterDims4D.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterDims4D is empty!"), return FAILED);

  ret = Calc4DWeightAvgPool(filterDims4D, kernelDataCount, filterInt8Data, weightInt8Temp);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight failed."),
                    return ret);

  weightInt8OutParam = move(weightInt8Temp);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Successfully get weight for node %s.", opName.c_str());
  return SUCCESS;
}

Status AvgPoolV2FusionPass::DoBiasOptimizeAvgpool(ComputeGraph& graph, NodePtr poolingNode,
                                                  vector<NodePtr>& fusionNodes, const int64_t& ksizeH,
                                                  const int64_t& ksizeW, const int64_t& inputC) {
  FUSION_PASS_CHECK(poolingNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingNode is null, fusion failed."),
                    return PARAM_INVALID);

  OpDescPtr poolingOp = poolingNode->GetOpDesc();

  FUSION_PASS_CHECK(poolingOp == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingOp is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "quant bias optimize op %s, begin to bias optimize.", poolingOp->GetName().c_str());

  // get offsetA from poolingOp
  int32_t offsetA = 0;
  (void)AttrUtils::GetInt(poolingOp, ATTR_OFFSET_X, offsetA);

  offsetA = static_cast<int8_t>(offsetA);

  /* Get pooling Weight filter */
  vector<GeTensorPtr> weights_pooling = OpDescUtils::MutableWeights(poolingNode);
  FUSION_PASS_CHECK(weights_pooling.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weights_pooling is null ptr!"),
                    return PARAM_INVALID);
  GeTensorPtr filter = weights_pooling[0];
  FUSION_PASS_CHECK(filter == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filter is null ptr!"),
                    return PARAM_INVALID);
  int8_t *filterInt8Data = (int8_t *)(filter->GetData().data());
  vector<int64_t> filterDims = {1, inputC, ksizeH, ksizeW};

  /* Store the filter data after optimization */
  unique_ptr<int32_t[]> weightInt8OutParam;
  Status ret = GetWeightOfConvAvgpool(poolingNode->GetName(), filterInt8Data, filterDims, weightInt8OutParam);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight of conv failed."),
                    return ret);
  int64_t co = filterDims[INDEX_CO_avg];

  // do not have bias, create bias node and init bias
  // And in this case, we do not need to shared the bias with other conv
  // So just create a new bias and set the data.

  OP_LOGD(FUSED_OP_TYPE.c_str(), "cube [%s] has no bias, create bias and set data", poolingNode->GetName().c_str());
  OP_LOGD(FUSED_OP_TYPE.c_str(), "the cube node have %ld in data Anchors", poolingNode->GetAllInDataAnchors().size());

  // set bias
  GeTensorDesc tmpDesc;
  GeTensorPtr biasPtr = nullptr;

  unique_ptr<int32_t[]> biasDataTemp(new (nothrow) int32_t[co]());
  FUSION_PASS_CHECK(biasDataTemp == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "biasDataTemp is nullptr"),
                    return PARAM_INVALID);
  for (int64_t i = 0; i < co; i++) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "offset[%d] before %s ", i, to_string(offsetA).c_str());
    OP_LOGD(FUSED_OP_TYPE.c_str(), "weight[%d] before %d ", i, weightInt8OutParam[i]);
    int64_t isaArchVer = 0;
    AttrUtils::GetInt(poolingOp, "isaArchVer", isaArchVer);
    if (isaArchVer == 1) {
      biasDataTemp[i] = 0;
    } else {
      biasDataTemp[i] = -offsetA * weightInt8OutParam[i];
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "bias [%d] after %d ", i, biasDataTemp[i]);
  }

  FUSION_PASS_MAKE_SHARED(
      biasPtr = make_shared<GeTensor>(tmpDesc, (uint8_t *)(biasDataTemp.get()), co * sizeof(int32_t)),
      biasPtr = nullptr;
      return PARAM_INVALID);

  // update weights
  GeShape biasShape({co});
  biasPtr->MutableTensorDesc().SetShape(biasShape);
  biasPtr->MutableTensorDesc().SetDataType(DT_INT32);

  ret = biasPtr->SetData(reinterpret_cast<uint8_t *>(biasDataTemp.get()), co * sizeof(int32_t));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "set bias data failed!"), return ret);

  FUSION_PASS_CHECK(
      PatternFusionUtil::SetWeightByIndex(poolingNode, biasPtr, 2, graph) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(poolingNode->GetName().c_str(), "Fail to add bias const node for pooling node."),
      return FAILED);

  // update the bias outputDesc of biasOpDesc
  GeTensorDesc inputDesc0 = poolingOp->GetInputDesc(0);
  Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  int biasInputIndex = 2;
  auto biasPeerOutAnchor = poolingNode->GetInDataAnchor(biasInputIndex)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(biasPeerOutAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "biasPeerOutAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr biasNode = poolingNode->GetInDataAnchor(biasInputIndex)->GetPeerOutAnchor()->GetOwnerNode();
  OpDescPtr biasOpDesc = biasNode->GetOpDesc();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "bias_node_name is %s", biasNode->GetName().c_str());

  // only has one output, index 0
  GeTensorDesc biasOutputDesc = biasOpDesc->GetOutputDesc(0);
  biasOutputDesc.SetShape(biasShape);
  biasOutputDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasOutputDesc.SetOriginShape(biasShape);
  biasOutputDesc.SetOriginDataType(DT_INT32);
  biasOutputDesc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(
      biasOpDesc->UpdateOutputDesc(0, biasOutputDesc) != GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update output desc of BiasNode[%s] not success.",
                                     biasOpDesc->GetName().c_str()),
      return FAILED);

  // update the bias inputDesc of the convOpDesc
  GeTensorDesc biasDesc = poolingOp->GetInputDesc(biasInputIndex);
  biasDesc.SetShape(biasShape);
  biasDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasDesc.SetOriginShape(biasShape);
  biasDesc.SetOriginDataType(DT_INT32);
  biasDesc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(
      poolingOp->UpdateInputDesc(2, biasDesc) != GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update bias input desc of ConvNode[%s] not success.",
                                     poolingNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

Status AvgPoolV2FusionPass::UpdateDequantConst(const NodePtr &const_node, const float &area_factor) const {
  vector<GeTensorPtr> const_dequant = OpDescUtils::MutableWeights(const_node);
  FUSION_PASS_CHECK(const_dequant.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_dequant is null ptr!"),
                    return PARAM_INVALID);
  GeTensorPtr const_ptr = const_dequant[0];
  FUSION_PASS_CHECK(const_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_ptr is null ptr!"),
                    return PARAM_INVALID);
  float *const_data = (float *)(const_ptr->GetData().GetData());
  FUSION_PASS_CHECK(const_data == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data is null ptr!"),
                    return PARAM_INVALID);
  const_data[0] = area_factor * const_data[0];
  OP_LOGD(FUSED_OP_TYPE.c_str(), "const_data is %f", const_data[0]);
  const_ptr->SetData(reinterpret_cast<uint8_t *>(const_data), 2 * sizeof(float));
  return SUCCESS;
}

Status AvgPoolV2FusionPass::Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &fusionNodes) {
  // avgpool node
  NodePtr avgPoolNode = GetNodeFromMapping(PATTERN_AVGPOOL, mapping);
  FUSION_PASS_CHECK(avgPoolNode == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolV2Node is null, fusion failed."),
                    return PARAM_INVALID);

  // input of AvgPool
  OpDescPtr avgPoolDesc = avgPoolNode->GetOpDesc();
  FUSION_PASS_CHECK(avgPoolDesc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolV2Node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  GeTensorDesc avgPoolInputTensor = avgPoolNode->GetOpDesc()->GetInputDesc(0);

  // get shape
  GeShape avgPoolInputShape = avgPoolInputTensor.GetShape();
  Format inputOriginFormat = avgPoolInputTensor.GetOriginFormat();
  // geshape->vector
  vector<int64_t> dimInfo = avgPoolInputShape.GetDims();
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  bool isDynamic = false;
  // when static op or dynamic op phase_running, is_dynamic = false
  if (find(dimInfo.begin(), dimInfo.end(), -1) != dimInfo.end()) {
    isDynamic = true;
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
  vector<int64_t> padv3_pads;
  // get windowsize padding strides value dataFormat
  AttrUtils::GetStr(avgPoolDesc, "data_format", dataFormat);
  AttrUtils::GetStr(avgPoolDesc, "padding_mode", padding);
  AttrUtils::GetListInt(avgPoolDesc, "ksize", ksize);
  AttrUtils::GetListInt(avgPoolDesc, "strides", strides);
  AttrUtils::GetBool(avgPoolDesc, "global_pooling", global_pooling);
  AttrUtils::GetBool(avgPoolDesc, "ceil_mode", ceil_mode);
  AttrUtils::GetBool(avgPoolDesc, "exclusive", exclusive);
  AttrUtils::GetListInt(avgPoolDesc, "pads", pads);
  AttrUtils::GetListInt(avgPoolDesc, "_padv3_pads", padv3_pads);
  if (!padv3_pads.empty()) {
    // if enable padv3+avgpool fusion, need to remove pad from padv3
    if (padv3_pads.size() != 4) {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "padv3_pads size should be same as pads.");
      return NOT_CHANGED;
    }
    for (int pads_idx = 0; pads_idx < pads.size(); ++pads_idx) {
      pads[pads_idx] = pads[pads_idx] - padv3_pads[pads_idx];
    }
    if (inputOriginFormat == FORMAT_NHWC) {
      dimInfo[1] = dimInfo[1] + padv3_pads[0] + padv3_pads[1];
      dimInfo[2] = dimInfo[2] + padv3_pads[2] + padv3_pads[3];
    } else if (inputOriginFormat == FORMAT_NCHW) {
      dimInfo[2] = dimInfo[2] + padv3_pads[0] + padv3_pads[1];
      dimInfo[3] = dimInfo[3] + padv3_pads[2] + padv3_pads[3];
    }
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
  if (stridesH > 63 || stridesW > 63) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "strided_h or strided_w >63, not support");
    return NOT_CHANGED;
  }
  bool AicoreSupport = true;
  AicoreSupport = CheckOpSupported(avgPoolDesc);
  if (!isDynamic && !AicoreSupport) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "aicore not support");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(!AttrUtils::SetInt(avgPoolNode->GetOpDesc(), "groups", inputC),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "set groups attr failed"), return FAILED);
  // get pre node of pooling
  InDataAnchorPtr poolingAnchorPtr0 = avgPoolNode->GetInDataAnchor(0);
  OutDataAnchorPtr preAnchorPtr0 = poolingAnchorPtr0->GetPeerOutAnchor();
  NodePtr preNode = preAnchorPtr0->GetOwnerNode();

  bool IsInt8 = false;
  if (avgPoolNode->GetOpDesc()->GetInputDesc(0).GetDataType() == DT_INT8) {
    IsInt8 = true;
  }

  if (preNode->GetOpDesc()->GetType() == "AscendQuant" || IsInt8) {
    // int8
    // if pooling`s pre op is AscendQuant, pooling assitMatrix dtype is int8, and need add bias for pooling node
    NodePtr dequantNode = nullptr;
    if (avgPoolNode->GetAllOutDataAnchors().empty()) {
      return PARAM_INVALID;
    }

    for (OutDataAnchorPtr outDataAnchor : avgPoolNode->GetAllOutDataAnchors()) {
      if (outDataAnchor == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", avgPoolNode->GetName().c_str());
        return PARAM_INVALID;
      }
      if (outDataAnchor->GetPeerInDataAnchors().empty()) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", avgPoolNode->GetName().c_str());
        return PARAM_INVALID;
      }
      for (InDataAnchorPtr inDataAnchorPtr : outDataAnchor->GetPeerInDataAnchors()) {
        if (inDataAnchorPtr == nullptr) {
          OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr in Data anchor.", avgPoolNode->GetName().c_str());
          return PARAM_INVALID;
        }
        dequantNode = inDataAnchorPtr->GetOwnerNode();
        if ((dequantNode->GetType() == "AscendDequant") || (dequantNode->GetType() == "AscendRequant")) {
          break;
        }
      }
    }
    if (dequantNode == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolNode does not have a dequantNode output node.");
      return FAILED;
    }

    if (dequantNode->GetAllInDataAnchors().empty()) {
      return PARAM_INVALID;
    }
    InDataAnchorPtr inDataAnchorPtr1 = dequantNode->GetInDataAnchor(1);
    if (inDataAnchorPtr1 == nullptr) {
      return PARAM_INVALID;
    }
    OutDataAnchorPtr outDataAnchorPtr = inDataAnchorPtr1->GetPeerOutAnchor();
    if (outDataAnchorPtr == nullptr) {
      return PARAM_INVALID;
    }

    NodePtr HostcpuNode = outDataAnchorPtr->GetOwnerNode();
    string type = NodeUtils::GetInConstNodeTypeCrossSubgraph(HostcpuNode);
    if ((HostcpuNode->GetType() != "RequantHostCpuOp") && (type != "Const") &&
        (HostcpuNode->GetType() != "RequantHostCpuOpV2")) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "dont find op RequantHostCpuOp or not find const");
      return NOT_CHANGED;
    }

    int64_t matrixSize = inputC * 1 * ksizeH * ksizeW;
    FUSION_PASS_CHECK(matrixSize <= 0, OP_LOGW(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"), return NOT_CHANGED);
    unique_ptr<int8_t[]> inputAssitInt8(new (nothrow) int8_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssitInt8.get() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitInt8 is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(matrixSize, INT8_NUM_ONE, *reinterpret_cast<int8_t *>(inputAssitInt8.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                      return ret);

    vector<int64_t> assitDimInfoOrigin = {inputC, 1, ksizeH, ksizeW};
    if (!exclusive || padding == "VALID") {
      float areaFactor = 1.0 / (ksizeH * ksizeW);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "padding is VALID, or exclusive is false!");
      // generate one matrix
      if (HostcpuNode->GetType() == "RequantHostCpuOp" || HostcpuNode->GetType() == "RequantHostCpuOpV2") {
        FUSION_PASS_CHECK(!AttrUtils::SetStr(HostcpuNode->GetOpDesc(), "padding", padding),
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
        FUSION_PASS_CHECK(!AttrUtils::SetFloat(HostcpuNode->GetOpDesc(), "area_factor", areaFactor),
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "Set area_factor attr failed."), return FAILED);
      } else {
        FUSION_PASS_CHECK(UpdateDequantConst(HostcpuNode, areaFactor) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update dequant const failed."),
                          return FAILED);
      }
    } else if (padding == "SAME") {
      // judge for unknown shape
      int64_t mulC = 0;
      GeTensorDesc input_desc = dequantNode->GetOpDesc()->GetOutputDesc(0);
      vector<int64_t> dimMul = input_desc.GetShape().GetDims();
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
      NodePtr mulNode = AddMul(graph, dequantNode, inputOriginFormat);
      FUSION_PASS_CHECK(mulNode == nullptr,
                        CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      // judge input dims for unknown shape
      GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      GeShape outputShape = inputDesc0.GetOriginShape();
      vector<int64_t> dimOut = outputShape.GetDims();
      for (size_t i = 0; i <= 3; i++) {
        auto dim = dimOut[i];
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      FUSION_PASS_CHECK(AddCoffe(mulNode, padding, dimInfo, window, stride) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
    } else if (padding == "CALCULATED") {
      // judge for unknown shape
      int64_t mulC = 0;
      GeTensorDesc input_desc = dequantNode->GetOpDesc()->GetOutputDesc(0);
      vector<int64_t> dimMul = input_desc.GetShape().GetDims();
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
      NodePtr mulNode = AddMul(graph, dequantNode, inputOriginFormat);
      FUSION_PASS_CHECK(mulNode == nullptr,
                        CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      // judge input dims for unknown shape
      GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      GeShape outputShape = inputDesc0.GetOriginShape();
      vector<int64_t> dimOut = outputShape.GetDims();
      for (size_t i = 0; i <= 3; i++) {
        auto dim = dimOut[i];
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "AvgPoolV2FusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      FUSION_PASS_CHECK(AddCoffe(mulNode, padding, dimInfo, window, stride, pads) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "padding is wrong, please check!");
      return PARAM_INVALID;
    }
    GeTensorDesc tensorDesc;
    GeShape assitShape(assitDimInfoOrigin);
    GeShape assitShapeOrigin(assitDimInfoOrigin);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetDataType(DT_INT8);
    tensorDesc.SetFormat(FORMAT_NCHW);
    tensorDesc.SetOriginFormat(FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOrigin);

    GeTensorPtr assitPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = make_shared<GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssitInt8.get()),
                                          matrixSize * sizeof(int8_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
    vector<GeTensorPtr> weights = {assitPtr};
    OpDescUtils::SetWeights(avgPoolNode, weights);
    auto constInputNodes = OpDescUtils::GetConstInputs(avgPoolNode);
    NodePtr constInput = nullptr;
    if (constInputNodes.size() != 0) {
      constInput = constInputNodes[0];
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return PARAM_INVALID;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOP);
    // add bias for pooling node
    ret = DoBiasOptimizeAvgpool(graph, avgPoolNode, fusionNodes, ksizeH, ksizeW, inputC);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "do fusion failed!"),
                      return ret);
  } else {
    int64_t matrixSize = inputC * 1 * ksizeH * ksizeW;
    int64_t inputC1 = (inputC + COUT - 1) / COUT;
    if (isDynamic) {
      matrixSize = inputC1 * ksizeH * ksizeW * CIN * COUT;
    }
    FUSION_PASS_CHECK(matrixSize <= 0, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"),
                      return PARAM_INVALID);

    unique_ptr<uint16_t[]> inputAssit(new (nothrow) uint16_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret;
    ret = NnSet(matrixSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(inputAssit.get()));
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
      FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."),
                        return ret);
    } else if (padding == "SAME") {
      float areaFactor = 1.0;
      // generate one matrix
      if (!isDynamic) {
        ret = GenerateFilterFP16V2(matrixSize, areaFactor, *inputAssit.get());
      } else {
        ret = GenerateFilterFP16DynamicV2(assitDimInfoDynamic, areaFactor, *inputAssit.get());
      }
      FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."),
                        return ret);
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
        NodePtr mulNode = AddMul(graph, avgPoolNode, inputOriginFormat);
        FUSION_PASS_CHECK(mulNode == nullptr,
                          CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                          return PARAM_INVALID);
        // judge input dims for unknown shape
        GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
        GeShape outputShape = inputDesc0.GetOriginShape();
        vector<int64_t> dimOut = outputShape.GetDims();
        for (size_t i = 0; i <= 3; i++) {
          auto dim = dimOut[i];
          if (PatternFusionUtil::IsUnknownShape(dim)) {
            OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolV2FusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
        FUSION_PASS_CHECK(AddCoffe(mulNode, padding, dimInfo, window, stride) != SUCCESS,
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
      FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."),
                        return ret);
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
        NodePtr mulNode = AddMul(graph, avgPoolNode, inputOriginFormat);
        FUSION_PASS_CHECK(mulNode == nullptr,
                          CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                          return PARAM_INVALID);
        // judge input dims for unknown shape
        GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
        GeShape outputShape = inputDesc0.GetOriginShape();
        vector<int64_t> dimOut = outputShape.GetDims();
        for (size_t i = 0; i <= 3; i++) {
          auto dim = dimOut[i];
          if (PatternFusionUtil::IsUnknownShape(dim)) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "AvgPoolV2FusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
        FUSION_PASS_CHECK(AddCoffe(mulNode, padding, dimInfo, window, stride, pads) != SUCCESS,
                          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
      }
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "padding is wrong, please check!");
      return PARAM_INVALID;
    }
    FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16V2 failed."),
                      return ret);
    GeTensorDesc tensorDesc;
    GeShape assitShape(assitDimInfoOrigin);
    GeShape assitShapeOrigin(assitDimInfoOrigin);
    GeShape assitShapeDynamic(assitDimInfoDynamic);
    GeShape assitShapeOriginDynamic(assitDimInfoOriginDynamic);
    if (!isDynamic) {
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetDataType(DT_FLOAT16);
      tensorDesc.SetFormat(FORMAT_NCHW);
      tensorDesc.SetOriginFormat(FORMAT_NCHW);
      tensorDesc.SetOriginShape(assitShapeOrigin);
      tensorDesc.SetOriginDataType(DT_FLOAT16);
    } else {
      tensorDesc.SetShape(assitShapeDynamic);
      tensorDesc.SetDataType(DT_FLOAT16);
      tensorDesc.SetFormat(FORMAT_FRACTAL_Z);
      tensorDesc.SetOriginFormat(FORMAT_NCHW);
      tensorDesc.SetOriginShape(assitShapeOriginDynamic);
      tensorDesc.SetOriginDataType(DT_FLOAT16);
    }

    GeTensorPtr assitPtr = nullptr;
    FUSION_PASS_MAKE_SHARED((assitPtr = make_shared<GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                              matrixSize * sizeof(uint16_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights = {assitPtr};
    OpDescUtils::SetWeights(avgPoolNode, weights);
    auto constInputNodes = OpDescUtils::GetConstInputs(avgPoolNode);
    NodePtr constInput = nullptr;
    if (constInputNodes.size() != 0) {
      constInput = constInputNodes[0];
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return PARAM_INVALID;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOP);
  }
  fusionNodes.push_back(avgPoolNode);
  return SUCCESS;
}

REGISTER_PASS("AvgPoolV2FusionPass", BUILT_IN_GRAPH_PASS, AvgPoolV2FusionPass);
}  // namespace fe
