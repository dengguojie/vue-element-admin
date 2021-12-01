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
 * \file avg_pool_pass.cpp
 * \brief avgPool fusion pass
 */
#include "avg_pool_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "quant_host_cpu_op_common.h"
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

using namespace std;
using namespace ge;

namespace fe {
static const uint16_t UINT_NUM_ZERO = 0;
static const int8_t INT8_NUM_ZERO = 0;
static const int8_t INT8_NUM_ONE = 1;
static const int32_t INT_NUM_FOUR = 4;
static const string PATTERN_AVGPOOL = "AvgPool";
static const std::string CONSTANTOP = "Const";
static const char* AVGPOOL = "AvgPool";
static const int64_t COUT = 16;
static const int64_t CIN = 16;
const int32_t INDEX_CO_avg = 1;
const int32_t INDEX_CI_avg = 0;
// kernel_h*kernel_w
static const int64_t AVG_KERNEL_SIZE_H_MUL_W = 255;
// ksize restrictions
static const int64_t AVG_KERNEL_SIZE = 20;

Status GenerateFilterFP16(const int64_t& size, const float& areaFactor, uint16_t& output1) {
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

Status GenerateCoffeFP16(const vector<int64_t> shape, vector<int64_t> window, vector<int64_t> stride,
                         vector<int64_t> pad, const int64_t dimH, const int64_t dimW, uint16_t& output1) {
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
            fp16_t out_val;
            out_val.val = 0;
            out_val = (float)area;
            output[m * (shape[1] * shape[2] * shape[3] * shape[4]) + n * (shape[2] * shape[3] * shape[4]) +
                   i * (shape[3] * shape[4]) + j * shape[4] + k] = out_val.val;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status GenerateFilterFP16Dynamic(const vector<int64_t> shape, const float areaFactor, uint16_t& output1) {
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

NodePtr AvgPoolFusionPass::AddMul(ge::ComputeGraph& graph, ge::NodePtr& avgPoolNode, ge::Format& inputOriginFormat) {
  ge::OutDataAnchorPtr avgPoolAnchorPtr1 = avgPoolNode->GetOutDataAnchor(0);
  ge::NodePtr postNode = nullptr;
  ge::NodePtr mulNode = nullptr;
  int64_t mulN = 0;
  int64_t mulH = 0;
  int64_t mulW = 0;
  int64_t mulC = 0;
  int64_t mulC1 = 0;

  // creat a antiquant node
  std::shared_ptr<ge::OpDesc> mulDesc = nullptr;
  mulDesc = std::make_shared<ge::OpDesc>(avgPoolNode->GetName() + "_mul_layer", "Mul");
  FUSION_PASS_CHECK(mulDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulDesc is null, mul failed."), return nullptr);

  // add input
  ge::GeTensorDesc input_desc = avgPoolNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape mulShape = input_desc.GetShape();
  vector<int64_t> dimMul = mulShape.GetDims();

  if (dimMul.size() != 0) {
    if (inputOriginFormat == FORMAT_NHWC) {
      mulN = dimMul[0];
      mulH = dimMul[1];
      mulW = dimMul[2];
      mulC = dimMul[3];
    }
    else if (inputOriginFormat == FORMAT_NCHW) {
      mulN = dimMul[0];
      mulH = dimMul[2];
      mulW = dimMul[3];
      mulC = dimMul[1];
    }
    else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputOriginFormat only support NHWC and NCHW");
      return nullptr;
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimMul is null, please check!");
    return nullptr;
  }

  mulC1 = (mulC + COUT - 1) / COUT;
  vector<int64_t> mulDimInfo = {mulN, mulC1, mulH, mulW, 16};

  ge::GeShape mulInputShape(mulDimInfo);
  input_desc.SetShape(mulInputShape);
  input_desc.SetOriginShape(mulShape);
  input_desc.SetFormat(ge::FORMAT_NC1HWC0);
  input_desc.SetOriginFormat(inputOriginFormat);
  input_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_CHECK(mulDesc->AddInputDesc(input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc input failed."), return nullptr);

  // add output
  ge::GeTensorDesc output_desc;
  ge::GeShape mulOutputShape(mulDimInfo);
  output_desc.SetShape(mulOutputShape);
  output_desc.SetOriginShape(mulShape);
  output_desc.SetFormat(ge::FORMAT_NC1HWC0);
  output_desc.SetOriginFormat(inputOriginFormat);
  output_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_CHECK(mulDesc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc output failed."), return nullptr);

  // add node
  mulNode = graph.AddNode(mulDesc);
  FUSION_PASS_CHECK(mulNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return nullptr);

  for (auto postAnchorPtr0 : avgPoolAnchorPtr1->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(postAnchorPtr0 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "postAnchorPtr0 is null, fusion failed."),
                      return nullptr);

    postNode = postAnchorPtr0->GetOwnerNode();

    // remove edge between avgpool and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, avgPoolAnchorPtr1) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              mulNode->GetName().c_str(), postNode->GetName().c_str()),
                      return nullptr);
  }
  // add edge between avgpool and mul
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(avgPoolAnchorPtr1, mulNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            avgPoolNode->GetName().c_str(), mulNode->GetName().c_str()),
                    return nullptr);

  return mulNode;
}

Status AvgPoolFusionPass::AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mulNode, string& padding,
                                   vector<int64_t>& dimInfo, vector<int64_t> ksize, vector<int64_t> stride) {
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
  pad = {0, 0, 0, 0};
  dilation = {1, 1};

  ge::OpDescPtr mulOp = mulNode->GetOpDesc();
  ge::GeTensorDesc inputDesc0 = mulOp->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  ge::GeShape outputShape = inputDesc0.GetOriginShape();
  vector<int64_t> dimOut = outputShape.GetDims();
  if (dimOut.size() != 0) {
    if (inputDesc0OriginFormat == FORMAT_NHWC) {
      outputH = dimOut[1];
      outputW = dimOut[2];
      outputC = dimOut[3];
      dimH = dimInfo[1];
      dimW = dimInfo[2];
    } else {
      outputH = dimOut[2];
      outputW = dimOut[3];
      outputC = dimOut[1];
      dimH = dimInfo[2];
      dimW = dimInfo[3];
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimOut is null, please check!");
    return PARAM_INVALID;
  }
  outputC1 = (outputC + outputC0 - 1) / outputC0;
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

  ge::GeTensorPtr coffePtr = nullptr;
  int64_t coffeSize = 1 * outputC1 * outputH * outputW * outputC0;
  FUSION_PASS_CHECK(coffeSize <= 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "coffeSize is Invalid"), return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[coffeSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret = NnSet(coffeSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

  vector<int64_t> coffeDimInfo = {1, outputC1, outputH, outputW, outputC0};
  ret = GenerateCoffeFP16(coffeDimInfo, ksize, stride, pad, dimH, dimW, *inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);

  vector<int64_t> coffeDimInfoOrigin;
  if (inputDesc0OriginFormat == FORMAT_NHWC) {
    coffeDimInfoOrigin = {1, outputH, outputW, outputC};
  } else if (inputDesc0OriginFormat == FORMAT_NCHW) {
    coffeDimInfoOrigin = {1, outputC, outputH, outputW};
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "format is wrong, please check!");
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
  FUSION_PASS_CHECK(mulDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {coffePtr};
  ge::OpDescUtils::SetWeights(mulNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(mulNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
    return PARAM_INVALID;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
}


Status AvgPoolFusionPass::Calc4DWeightAvgPool(const std::vector<int64_t>& filterDims4D, const int64_t& kernelDataCount,
                                              const int8_t* filterInt8Data,
                                              std::unique_ptr<int32_t[]>& weightInt8Temp) {
  FUSION_PASS_CHECK(filterDims4D.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterDims4D is empty!"), return FAILED);

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
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The index %ld is out of weightInt8Data's range", k),
                            return FAILED);
          sum_temp += filterInt8Data[k];
        }
      }
    }
    weightInt8Temp[j] = sum_temp;
  }
  return SUCCESS;
}

Status AvgPoolFusionPass::GetWeightOfConvAvgpool(const std::string& opName, const int8_t* filterInt8Data,
                                                 const std::vector<int64_t>& filterDims,
                                                 std::unique_ptr<int32_t[]>& weightInt8OutParam) {
  // get weight_int8
  int64_t min = 0;
  int64_t div = 4;
  std::vector<int64_t> filterDims4D;
  size_t sizeOfFilter = filterDims.size();
  for (uint32_t i = 0; i <= INDEX_FILTER_W; i++) {
    if (i < sizeOfFilter) {
      filterDims4D.emplace_back(filterDims.at(i));
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
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetkernelDataCount faild."), return PARAM_INVALID);

  FUSION_PASS_CHECK(filterInt8Data == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightInt8Data is nullptr"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount <= min, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "kernelDataCount is not a positive number."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount == min || kernelDataCount >= UINT_MAX / div,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "kernelDataCount is out of range."), return PARAM_INVALID);

  // calc weight: accumulate weights
  std::unique_ptr<int32_t[]> weightInt8Temp(new (std::nothrow) int32_t[filterDims4D[INDEX_CO_avg]]());
  FUSION_PASS_CHECK(weightInt8Temp == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightInt8Temp is nullptr"),
                    return PARAM_INVALID);
  Status ret;
  FUSION_PASS_CHECK(filterDims4D.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterDims4D is empty!"), return FAILED);

  ret = Calc4DWeightAvgPool(filterDims4D, kernelDataCount, filterInt8Data, weightInt8Temp);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight failed."), return ret);

  weightInt8OutParam = std::move(weightInt8Temp);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Successfully get weight for node %s.", opName.c_str());
  return SUCCESS;
}

Status AvgPoolFusionPass::DoBiasOptimizeAvgpool(ge::ComputeGraph& graph, ge::NodePtr poolingNode,
                                                vector<ge::NodePtr>& fusionNodes, int64_t&ksizeH,
                                                int64_t& ksizeW, int64_t& inputC) {
  FUSION_PASS_CHECK(poolingNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr poolingOp = poolingNode->GetOpDesc();

  FUSION_PASS_CHECK(poolingOp == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingOp is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "quant bias optimize op %s, begin to bias optimize.", poolingOp->GetName().c_str());

  // get offsetA from poolingOp
  int32_t offsetA = 0;
  (void)ge::AttrUtils::GetInt(poolingOp, ATTR_OFFSET_X, offsetA);

  offsetA = (int8_t)offsetA;

  /* Get pooling Weight filter */
  vector<ge::GeTensorPtr> weights_pooling = ge::OpDescUtils::MutableWeights(poolingNode);
  FUSION_PASS_CHECK(weights_pooling.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weights_pooling is null ptr!"),
                    return PARAM_INVALID);
  ge::GeTensorPtr filter = weights_pooling[0];
  FUSION_PASS_CHECK(filter == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filter is null ptr!"), return PARAM_INVALID);
  int8_t* filterInt8Data = (int8_t*)(filter->GetData().data());
  vector<int64_t> filterDims = {1, inputC, ksizeH, ksizeW};

  /* Store the filter data after optimization */
  std::unique_ptr<int32_t[]> weightInt8OutParam;
  Status ret = GetWeightOfConvAvgpool(poolingNode->GetName(), filterInt8Data, filterDims, weightInt8OutParam);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight of conv failed."), return ret);
  int64_t co = filterDims.at(INDEX_CO_avg);

  // do not have bias, create bias node and init bias
  // And in this case, we do not need to shared the bias with other conv
  // So just create a new bias and set the data.

  OP_LOGD(FUSED_OP_TYPE.c_str(), "cube [%s] has no bias, create bias and set data", poolingNode->GetName().c_str());
  OP_LOGD(FUSED_OP_TYPE.c_str(), "the cube node have %ld in data Anchors", poolingNode->GetAllInDataAnchors().size());

  // set bias
  ge::GeTensorDesc tmpDesc;
  ge::GeTensorPtr biasPtr = nullptr;

  std::unique_ptr<int32_t[]> biasDataTemp(new (std::nothrow) int32_t[co]());
  FUSION_PASS_CHECK(biasDataTemp == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "biasDataTemp is nullptr"),
                    return PARAM_INVALID);
  for (int64_t i = 0; i < co; i++) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "offset[%d] before %s ", i, std::to_string(offsetA).c_str());
    OP_LOGD(FUSED_OP_TYPE.c_str(), "weight[%d] before %d ", i, weightInt8OutParam[i]);
    int64_t isaArchVer = 0;
    ge::AttrUtils::GetInt(poolingOp, "isaArchVer", isaArchVer);
    if (isaArchVer == 1) {
      biasDataTemp[i] = 0;
    } else {
      biasDataTemp[i] = -offsetA * weightInt8OutParam[i];
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "bias [%d] after %d ", i, biasDataTemp[i]);
  }

  FUSION_PASS_MAKE_SHARED(
      biasPtr = std::make_shared<ge::GeTensor>(tmpDesc, (uint8_t*)(biasDataTemp.get()), co * sizeof(int32_t)),
      biasPtr = nullptr;
      return PARAM_INVALID);

  // update weights
  ge::GeShape biasShape({co});
  biasPtr->MutableTensorDesc().SetShape(biasShape);
  biasPtr->MutableTensorDesc().SetDataType(ge::DT_INT32);

  ret = biasPtr->SetData(reinterpret_cast<uint8_t*>(biasDataTemp.get()), co * sizeof(int32_t));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "set bias data failed!"), return ret);

  FUSION_PASS_CHECK(PatternFusionUtil::SetWeightByIndex(poolingNode, biasPtr, 2, graph) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(poolingNode->GetName().c_str(), "Fail to add bias const node for pooling node."),
                    return FAILED);

  // update the bias outputDesc of biasOpDesc
  ge::GeTensorDesc inputDesc0 = poolingOp->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  int biasInputIndex = 2;
  auto biasPeerOutAnchor = poolingNode->GetInDataAnchor(biasInputIndex)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(biasPeerOutAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "biasPeerOutAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  ge::NodePtr biasNode = poolingNode->GetInDataAnchor(biasInputIndex)->GetPeerOutAnchor()->GetOwnerNode();
  ge::OpDescPtr biasOpDesc = biasNode->GetOpDesc();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "bias_node_name is %s", biasNode->GetName().c_str());

  // only has one output, index 0
  ge::GeTensorDesc biasOutputDesc = biasOpDesc->GetOutputDesc(0);
  biasOutputDesc.SetShape(biasShape);
  biasOutputDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasOutputDesc.SetOriginShape(biasShape);
  biasOutputDesc.SetOriginDataType(ge::DT_INT32);
  biasOutputDesc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(
      biasOpDesc->UpdateOutputDesc(0, biasOutputDesc) != ge::GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update output desc of BiasNode[%s] not success.", biasOpDesc->GetName().c_str()),
      return FAILED);

  // update the bias inputDesc of the convOpDesc
  ge::GeTensorDesc biasDesc = poolingOp->GetInputDesc(biasInputIndex);
  biasDesc.SetShape(biasShape);
  biasDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasDesc.SetOriginShape(biasShape);
  biasDesc.SetOriginDataType(ge::DT_INT32);
  biasDesc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(poolingOp->UpdateInputDesc(2, biasDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update bias input desc of ConvNode[%s] not success.",
                            poolingNode->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

Status AvgPoolFusionPass::UpdateDequantConst(ge::ComputeGraph& graph, ge::NodePtr& const_node, float& area_factor) {
  vector<ge::GeTensorPtr> const_dequant = ge::OpDescUtils::MutableWeights(const_node);
  FUSION_PASS_CHECK(const_dequant.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_dequant is null ptr!"),
                    return PARAM_INVALID);
  ge::GeTensorPtr const_ptr = const_dequant[0];
  FUSION_PASS_CHECK(const_ptr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_ptr is null ptr!"),
                    return PARAM_INVALID);
  float* const_data = (float*)(const_ptr->GetData().GetData());
  FUSION_PASS_CHECK(const_data == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data is null ptr!"),
                    return PARAM_INVALID);
  const_data[0] = area_factor * const_data[0];
  OP_LOGD(FUSED_OP_TYPE.c_str(), "const_data is %f", const_data[0]);
  const_ptr->SetData(reinterpret_cast<uint8_t*>(const_data), 2 * sizeof(float));
  return SUCCESS;
}

vector<FusionPattern*> AvgPoolFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define AvgPoolFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AvgPoolFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_AVGPOOL, {AVGPOOL}).SetOutput(PATTERN_AVGPOOL);

  patterns.push_back(pattern);

  return patterns;
}

Status AvgPoolFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // avgpool node
  ge::NodePtr avgPoolNode = GetNodeFromMapping(PATTERN_AVGPOOL, mapping);
  FUSION_PASS_CHECK(avgPoolNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of AvgPool
  ge::OpDescPtr avgPoolDesc = avgPoolNode->GetOpDesc();
  FUSION_PASS_CHECK(avgPoolDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "avgPoolNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(!CheckOpSupported(avgPoolDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                      return NOT_CHANGED);
  ge::GeTensorDesc avgPoolInputTensor = avgPoolNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc avg_pool_output_tensor = avgPoolNode->GetOpDesc()->GetOutputDesc(0);
  // get shape
  ge::GeShape avgPoolInputShape = avgPoolInputTensor.GetShape();
  ge::GeShape avgPooloutputhape = avg_pool_output_tensor.GetShape();
  ge::Format inputOriginFormat = avgPoolInputTensor.GetOriginFormat();
  // GESHAPE->vector
  vector<int64_t> dimInfo = avgPoolInputShape.GetDims();
  vector<int64_t> out_dimInfo = avgPooloutputhape.GetDims();
  int64_t inputC = 0;
  int64_t output_w = 0;
  bool isDynamic = false;
  bool isFuzzBuild = false;
  ge::AttrUtils::GetBool(avgPoolDesc, ge::ATTR_NAME_FUZZ_BUILD, isFuzzBuild);
  // when static op or dynamic op phase_running, is_dynamic = false
  if (std::find(dimInfo.begin(),dimInfo.end(), -1) != dimInfo.end() || isFuzzBuild) {
    isDynamic = true;
  }
  if (dimInfo.size() == 4) {
    if (inputOriginFormat == FORMAT_NHWC) {
      inputC = dimInfo[3];
      output_w = out_dimInfo[2];
    } else if (inputOriginFormat == FORMAT_NCHW) {
      inputC = dimInfo[1];
      output_w = out_dimInfo[3];
    }
    if (PatternFusionUtil::IsUnknownShape(inputC)) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dimInfo is not right, please check!");
    return NOT_CHANGED;
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
  // get windowsize padding strides value dataFormat
  ge::AttrUtils::GetStr(avgPoolDesc, "data_format", dataFormat);
  ge::AttrUtils::GetStr(avgPoolDesc, "padding", padding);
  ge::AttrUtils::GetListInt(avgPoolDesc, "ksize", ksize);
  ge::AttrUtils::GetListInt(avgPoolDesc, "strides", strides);

  if (dataFormat == "NHWC") {
    if (ksize.size() == INT_NUM_FOUR and strides.size() == INT_NUM_FOUR) {
      ksizeH = ksize[1];
      ksizeW = ksize[2];
      stridesH = strides[1];
      stridesW = strides[2];
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ksize or strides is incorrect, please check!");
      return NOT_CHANGED;
    }
  } else if (dataFormat == "NCHW") {
    if (ksize.size() == INT_NUM_FOUR and strides.size() == INT_NUM_FOUR) {
      ksizeH = ksize[2];
      ksizeW = ksize[3];
      stridesH = strides[2];
      stridesW = strides[3];
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ksize or strides is incorrect, please check!");
      return NOT_CHANGED;
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dataFormat is wrong, please check!");
    return NOT_CHANGED;
  }
  window = {ksizeH, ksizeW};
  stride = {stridesH, stridesW};
  // judge global pooling or out_put_w==1
  if ((!isDynamic) && (output_w == 1)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "avgpool is global or output_w=1, graph not changed.");
    return NOT_CHANGED;
  }

  if(stridesH > 63 || stridesW > 63) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "strided_h or strided_w >63, not support");
    return NOT_CHANGED;
  }

  if (isDynamic && (ksizeH * ksizeW > AVG_KERNEL_SIZE_H_MUL_W)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ksize_h or ksize_w aicore not support when dynamic mode");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(avgPoolNode->GetOpDesc(), "groups", inputC),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set groups attr failed"),
                    return FAILED);
  // get pre node of pooling
  ge::InDataAnchorPtr poolingAnchorPtr0 = avgPoolNode->GetInDataAnchor(0);
  FUSION_PASS_CHECK(poolingAnchorPtr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingAnchorPtr0's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr preAnchorPtr0 = poolingAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();

  bool IsInt8 = false;
  if (avgPoolNode->GetOpDesc()->GetInputDesc(0).GetDataType() == ge::DT_INT8) {
    IsInt8 = true;
  }

  if (preNode->GetOpDesc()->GetType() == "AscendQuant" || IsInt8) {
    // int8
    // if pooling's pre op is AscendQuant, pooling assitMatrix dtype is int8, and need add bias for pooling node

    ge::NodePtr dequantNode = nullptr;
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
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(HostcpuNode);
    if ((HostcpuNode->GetType() != "RequantHostCpuOp") && (type != "Const") &&
    (HostcpuNode->GetType() != "RequantHostCpuOpV2")) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "dont find op RequantHostCpuOp or not find const");
      return NOT_CHANGED;
    }

    int64_t matrixSize = inputC * 1 * ksizeH * ksizeW;
    FUSION_PASS_CHECK(matrixSize <= 0, OP_LOGW(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"), return NOT_CHANGED);
    unique_ptr<int8_t[]> inputAssitInt8(new (std::nothrow) int8_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssitInt8.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitInt8 is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(matrixSize, INT8_NUM_ONE, *reinterpret_cast<int8_t*>(inputAssitInt8.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    vector<int64_t> assitDimInfoOrigin = {inputC, 1, ksizeH, ksizeW};

    if (padding == "SAME") {
      // judge input dims for unknown shape
      ge::GeTensorDesc input_desc = dequantNode->GetOpDesc()->GetOutputDesc(0);
      vector<int64_t> dimMul = input_desc.GetShape().GetDims();
      int64_t mulC = 0;
      if (dimMul.size() != 0) {
        if (inputOriginFormat == FORMAT_NHWC) {
          mulC = dimMul[3];
        }
        else if (inputOriginFormat == FORMAT_NCHW) {
          mulC = dimMul[1];
        }
        if (PatternFusionUtil::IsUnknownShape(mulC)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolFusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      ge::NodePtr mulNode = AddMul(graph, dequantNode, inputOriginFormat);
      FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      // judge input dims for unknown shape
      ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      ge::GeShape outputShape = inputDesc0.GetOriginShape();
      vector<int64_t> dimOut = outputShape.GetDims();
      for (size_t i = 1; i <= 3; i++) {
        auto dim = dimOut[i];
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "AvgPoolFusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
      }
      FUSION_PASS_CHECK(AddCoffe(graph, mulNode, padding, dimInfo, window, stride) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
    } else if (padding == "VALID") {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "padding is VALID");
      if (HostcpuNode->GetType() == "RequantHostCpuOp" || HostcpuNode->GetType() == "RequantHostCpuOpV2") {
        FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(HostcpuNode->GetOpDesc(), "padding", padding),
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
        float areaFactor = 1.0 / (ksizeH * ksizeW);
        FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(HostcpuNode->GetOpDesc(), "area_factor", areaFactor),
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "Set area_factor attr failed."), return FAILED);
      } else {
        float areaFactor = 1.0 / (ksizeH * ksizeW);
        FUSION_PASS_CHECK(UpdateDequantConst(graph, HostcpuNode, areaFactor) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update dequant const failed."), return FAILED);
      }

    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "padding is wrong, please check!");
      return NOT_CHANGED;
    }

    //  set const node shape
    GeTensorDesc tensorDesc;
    ge::GeShape assitShape(assitDimInfoOrigin);
    ge::GeShape assitShapeOrigin(assitDimInfoOrigin);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOrigin);
    ge::GeTensorPtr assitPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssitInt8.get()),
                                                   matrixSize * sizeof(int8_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
    vector<ge::GeTensorPtr> weights = {assitPtr};
    ge::OpDescUtils::SetWeights(avgPoolNode, weights);
    auto constInputNodes = OpDescUtils::GetConstInputs(avgPoolNode);
    NodePtr constInput = nullptr;
    if (constInputNodes.size() != 0) {
      constInput = constInputNodes[0];
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return PARAM_INVALID;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOP);
    // add bias for pooling node
    ret = DoBiasOptimizeAvgpool(graph, avgPoolNode, fusionNodes, ksizeH, ksizeW, inputC);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "do fusion failed!"), return ret);

  } else {  // fp16
    ge::GeTensorPtr assitPtr = nullptr;
    int64_t matrixSize = inputC * ksizeH * ksizeW;
    int64_t inputC1 = (inputC + COUT -1) / COUT;
    if (isDynamic) {
      matrixSize = inputC1 * ksizeH * ksizeW * CIN * COUT;
    }
    FUSION_PASS_CHECK(matrixSize <= 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"), return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(matrixSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
    vector<int64_t> assitDimInfoOrigin = {inputC, 1, ksizeH, ksizeW};
    vector<int64_t> assitDimInfoDynamic = {inputC1 * ksizeH * ksizeW, 1, CIN, COUT};
    vector<int64_t> assitDimInfoOriginDynamic = {inputC, 1, ksizeH, ksizeW};
    if (padding == "VALID") {
      float areaFactor = 1.0 / (ksizeH * ksizeW);
      // generate one matrix
      if (!isDynamic) {
        ret = GenerateFilterFP16(matrixSize, areaFactor, *inputAssit.get());
      } else {
        areaFactor = 1.0;
        ret = GenerateFilterFP16Dynamic(assitDimInfoDynamic, areaFactor, *inputAssit.get());
      }
    } else if (padding == "SAME") {
      float areaFactor = 1.0;
      // generate one matrix
      if (!isDynamic) {
        ret = GenerateFilterFP16(matrixSize, areaFactor, *inputAssit.get());
      } else {
        ret = GenerateFilterFP16Dynamic(assitDimInfoDynamic, areaFactor, *inputAssit.get());
      }
      if (!isDynamic) {
        // judge input dims for unknown shape
        ge::GeTensorDesc input_desc = avgPoolNode->GetOpDesc()->GetOutputDesc(0);
        ge::GeShape mulShape = input_desc.GetShape();
        vector<int64_t> dimMul = mulShape.GetDims();
        int64_t mulC = 0;
        if (dimMul.size() != 0) {
          if (inputOriginFormat == FORMAT_NHWC) {
            mulC = dimMul[3];
          }
          else if (inputOriginFormat == FORMAT_NCHW) {
            mulC = dimMul[1];
          }
          if (PatternFusionUtil::IsUnknownShape(mulC)) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AvgPoolFusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
        ge::NodePtr mulNode = AddMul(graph, avgPoolNode, inputOriginFormat);
        FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, AddMul failed."),
                          return PARAM_INVALID);
        // judge input dims for unknown shape
        ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
        ge::GeShape outputShape = inputDesc0.GetOriginShape();
        vector<int64_t> dimOut = outputShape.GetDims();
        for (size_t i = 1; i <= 3; i++) {
          auto dim = dimOut[i];
          if (PatternFusionUtil::IsUnknownShape(dim)) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AvgPoolFusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
        FUSION_PASS_CHECK(AddCoffe(graph, mulNode, padding, dimInfo, window, stride) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return ret);
      }
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "padding is wrong, please check!");
      return NOT_CHANGED;
    }

    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GenerateFilterFP16 failed."), return ret);

    ge::GeTensorDesc tensorDesc;
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
      OP_LOGW(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return NOT_CHANGED;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOP);
  }
  fusionNodes.push_back(avgPoolNode);
  return SUCCESS;
}

REGISTER_PASS("AvgPoolFusionPass", BUILT_IN_GRAPH_PASS, AvgPoolFusionPass);
}  // namespace fe
