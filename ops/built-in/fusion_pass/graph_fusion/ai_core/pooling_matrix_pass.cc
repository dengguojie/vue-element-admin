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
 * \file pooling_matrix_pass.cpp
 * \brief Pooling fusion pass
 */
#include "pooling_matrix_pass.h"

#include <utility>
#include <vector>
#include <string>
#include <map>
#include <cmath>

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

namespace fe {
static const uint16_t UINT_NUM_ZERO = 0;
static const int8_t INT8_NUM_ONE = 1;
static const char PATTERN_POOL[] = "Pooling";
static const char CONSTANTOPTAB[] = "Const";
static const char POOLINGTAB[] = "Pooling";
static const int64_t COUT = 16;
static const int64_t CIN = 16;
static const int64_t COUT32 = 32;
static const int64_t CIN32 = 32;

Status PoolingGenerateFilterFP16(const int64_t size, const float areaFactor, uint16_t& output1) {
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

Status IsPadZero(vector<int64_t> pad, bool& flag) {
  if (pad.size() != 4) {
    return FAILED;
  }
  flag = true;
  for (auto i : pad) {
    if (i != 0) {
      flag = false;
    }
  }
  return SUCCESS;
}

Status PoolingGenerateCoffeFP16(const vector<int64_t> shape, vector<int64_t> window, vector<int64_t> stride,
                         vector<int64_t> pad, const int64_t dimH, const int64_t dimW, uint16_t& output1, bool isInt8) {
  uint16_t* output = &output1;
  int64_t h_start = 0;
  int64_t w_start = 0;
  int64_t h_end = 0;
  int64_t w_end = 0;
  float area = 0;
  float base_area = window[0] * window[1];
  for (int m = 0; m < shape[0]; m++) {
    for (int n = 0; n < shape[1]; n++) {
      for (int64_t i = 0; i < shape[2]; i++) {
        for (int64_t j = 0; j < shape[3]; j++) {
          for (int k = 0; k < shape[4]; k++) {
            h_start = i * stride[0] - pad[0];
            w_start = j * stride[1] - pad[2];
            h_end = min(h_start + window[0], dimH + pad[0]);
            w_end = min(w_start + window[1], dimW + pad[2]);
            area = max((h_end - h_start) * (w_end - w_start), static_cast<int64_t>(1));
            area = 1.0 / area;
            if (isInt8) {
              area = area * base_area;
            }
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

NodePtr PoolingFusionPass::AddMul(ge::ComputeGraph& graph, ge::NodePtr& PoolNode) {
  ge::OutDataAnchorPtr PoolAnchorPtr1 = PoolNode->GetOutDataAnchor(0);
  ge::NodePtr postNode = nullptr;
  ge::NodePtr mulNode = nullptr;
  int64_t mulN = 0;
  int64_t mulH = 0;
  int64_t mulW = 0;
  int64_t mulC = 0;
  int64_t mulC1 = 0;

  // creat a mul node
  std::shared_ptr<ge::OpDesc> mulDesc = nullptr;
  mulDesc = std::make_shared<ge::OpDesc>(PoolNode->GetName() + "_mul_layer", "Mul");
  FUSION_PASS_CHECK(mulDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mulDesc is null, mul failed."),
                    return nullptr);

  // add input
  ge::GeTensorDesc input_desc = PoolNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape mulShape = input_desc.GetShape();
  vector<int64_t> dimMul = mulShape.GetDims();

  if (dimMul.size() >= 4) {
      mulN = dimMul[0];
      mulC = dimMul[1];
      mulH = dimMul[2];
      mulW = dimMul[3];
    } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimMul size is not match, please check!");
    return nullptr;
  }

  mulC1 = (mulC + 16 - 1) / 16;
  vector<int64_t> mulDimInfo = {mulN, mulC1, mulH, mulW, 16};

  ge::GeShape mulInputShape(mulDimInfo);
  input_desc.SetShape(mulInputShape);
  input_desc.SetOriginShape(mulShape);
  input_desc.SetFormat(ge::FORMAT_NC1HWC0);
  input_desc.SetOriginFormat(ge::FORMAT_NCHW);
  FUSION_PASS_CHECK(mulDesc->AddInputDesc(input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc input failed."), return nullptr);

  // add output
  ge::GeTensorDesc output_desc;
  ge::GeShape mulOutputShape(mulDimInfo);
  output_desc.SetShape(mulOutputShape);
  output_desc.SetOriginShape(mulShape);
  output_desc.SetFormat(ge::FORMAT_NC1HWC0);
  output_desc.SetOriginFormat(ge::FORMAT_NCHW);
  FUSION_PASS_CHECK(mulDesc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc output failed."),
                                                   return nullptr);

  // add node
  mulNode = graph.AddNode(mulDesc);

  for (auto postAnchorPtr0 : PoolAnchorPtr1->GetPeerInDataAnchors()) {
    postNode = postAnchorPtr0->GetOwnerNode();

    // remove edge between avgpool and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, PoolAnchorPtr1) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between node %s. and node %s failed.",
                              mulNode->GetName().c_str(), postNode->GetName().c_str()),
                      return nullptr);
  }
  // add edge between avgpool and mul
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(PoolAnchorPtr1, mulNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                            PoolNode->GetName().c_str(), mulNode->GetName().c_str()),
                    return nullptr);

  return mulNode;
}

Status PoolingFusionPass::AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mulNode, vector<int64_t> pad,
                                   vector<int64_t>& dimInfo, vector<int64_t> window, vector<int64_t> stride,
                                   std::string& recode, bool isInt8) {
  int64_t outputH = 0;
  int64_t outputW = 0;
  int64_t outputC = 0;
  int64_t outputC1 = 0;
  int64_t outputC0 = 16;
  int64_t dimH = 0;
  int64_t dimW = 0;

  ge::OpDescPtr mulOp = mulNode->GetOpDesc();
  ge::GeTensorDesc inputDesc0 = mulOp->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  ge::GeShape outputShape = inputDesc0.GetOriginShape();
  vector<int64_t> dimOut = outputShape.GetDims();
  if (dimOut.size() >= 4) {
      outputH = dimOut[2];
      outputW = dimOut[3];
      outputC = dimOut[1];
      dimH = dimInfo[2];
      dimW = dimInfo[3];
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimOut size is not match, please check!");
    return PARAM_INVALID;
  }
  outputC1 = (outputC + outputC0 - 1) / outputC0;
  ge::GeTensorPtr coffePtr = nullptr;
  int64_t coffeSize = 1 * outputC1 * outputH * outputW * outputC0;
  FUSION_PASS_CHECK(coffeSize <= 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "coffeSize is Invalid"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[coffeSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret = NnSet(coffeSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

  vector<int64_t> coffeDimInfo = {1, outputC1, outputH, outputW, outputC0};
  ret = PoolingGenerateCoffeFP16(coffeDimInfo, window, stride, pad, dimH, dimW, *inputAssit.get(), isInt8);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."),
                    return ret);

  vector<int64_t> coffeDimInfoOrigin;
  if (inputDesc0OriginFormat == FORMAT_NCHW) {
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
  ge::OpDescPtr constOpdesc = ge::OpDescUtils::CreateConstOp(coffePtr);
  FUSION_PASS_CHECK(constOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to create const op desc."),
                    return FAILED);
  ge::NodePtr constNode = nullptr;
  if (recode.empty() or !mulConstNode.count(recode)) {
    constNode = graph.AddNode(constOpdesc);
    if (!recode.empty()) {
      mulConstNode[recode] = constNode;
    }
  }
  else {
    unordered_map<string, ge::NodePtr> ::iterator iter;
    iter = mulConstNode.find(recode);
    if (iter != mulConstNode.end()) {
      constNode = iter->second;
    }
  }

  FUSION_PASS_CHECK(constNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add const node."),
                    return FAILED);
  FUSION_PASS_CHECK(mulNode->AddLinkFrom(1, constNode) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to link const node with pooling node."),
                    return FAILED);

  ge::OpDescPtr mulDesc = mulNode->GetOpDesc();
  FUSION_PASS_CHECK(mulDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mulNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  auto constInputNodes = OpDescUtils::GetConstInputs(mulNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
    return PARAM_INVALID;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOPTAB);

  return SUCCESS;
}

Status PoolingFusionPass::Calc4DWeight(const std::vector<int64_t>& filterDims4D, const int64_t& kernelDataCount,
                                       const int8_t* filterInt8Data, std::unique_ptr<int32_t[]>& weightInt8Temp) {
  FUSION_PASS_CHECK(filterDims4D.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "filterDims4D is empty!"),
                    return FAILED);
  for (int64_t j = 0; j < filterDims4D[INDEX_CO]; j++) {
    int64_t sum_temp = 0;
    for (int64_t i = 0; i < filterDims4D[INDEX_CI]; i++) {
      for (int64_t h = 0; h < filterDims4D[INDEX_FILTER_H]; h++) {
        for (int64_t w = 0; w < filterDims4D[INDEX_FILTER_W]; w++) {
          int64_t k = (j * filterDims4D[INDEX_CI] * filterDims4D[INDEX_FILTER_H] * filterDims4D[INDEX_FILTER_W]) +
                      (i * filterDims4D[INDEX_FILTER_H] * filterDims4D[INDEX_FILTER_W]) +
                      (h * filterDims4D[INDEX_FILTER_W]) + w;
          FUSION_PASS_CHECK(k >= kernelDataCount,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                           "The index %ld is out of weightInt8Data's range",
                                                           k),
                            return FAILED);
          sum_temp += filterInt8Data[k];
        }
      }
    }
    weightInt8Temp[j] = sum_temp;
  }
  return SUCCESS;
}

Status PoolingFusionPass::GetWeightOfConv(const std::string& opName, const int8_t* filterInt8Data,
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
            filterDims4D[INDEX_CO], filterDims4D[INDEX_CI], filterDims4D[INDEX_FILTER_H], filterDims4D[INDEX_FILTER_W]);
  }

  // get conv core kerneldata count
  int64_t kernelDataCount = 1;
  FUSION_PASS_CHECK(GetkernelDataCountForPass(filterDims, kernelDataCount) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetkernelDataCount faild."),
                                                   return PARAM_INVALID);

  FUSION_PASS_CHECK(filterInt8Data == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "weightInt8Data is nullptr"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount <= min, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "kernelDataCount is not a positive number."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kernelDataCount == min || kernelDataCount >= UINT_MAX / div,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "kernelDataCount is out of range."),
                                                   return PARAM_INVALID);

  // calc weight: accumulate weights
  std::unique_ptr<int32_t[]> weightInt8Temp(new (std::nothrow) int32_t[filterDims4D[INDEX_CO]]());
  FUSION_PASS_CHECK(weightInt8Temp == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "weightInt8Temp is nullptr"),
                    return PARAM_INVALID);
  Status ret;
  FUSION_PASS_CHECK(filterDims4D.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "filterDims4D is empty!"),
                    return FAILED);

  ret = Calc4DWeight(filterDims4D, kernelDataCount, filterInt8Data, weightInt8Temp);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight failed."),
                    return ret);

  weightInt8OutParam = std::move(weightInt8Temp);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Successfully get weight for node %s.", opName.c_str());
  return SUCCESS;
}

Status PoolingFusionPass::DoBiasOptimize(ge::ComputeGraph& graph, ge::NodePtr poolingNode,
                                         vector<ge::NodePtr>& fusionNodes,int64_t& windowH,
                                         int64_t& windowW, int64_t& inputC) {
  FUSION_PASS_CHECK(poolingNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "poolingNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr poolingOp = poolingNode->GetOpDesc();

  FUSION_PASS_CHECK(poolingOp == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "poolingOp is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "quant bias optimize op %s, begin to bias optimize.", poolingOp->GetName().c_str());

  // get offsetA from poolingOp
  int32_t offsetA = 0;
  (void)ge::AttrUtils::GetInt(poolingOp, ATTR_OFFSET_X, offsetA);
  offsetA = (int8_t)(offsetA);

  /* Get pooling Weight filter */
  vector<ge::GeTensorPtr> weights_pooling = ge::OpDescUtils::MutableWeights(poolingNode);
  FUSION_PASS_CHECK(weights_pooling.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "weights_pooling is nullptr!"),
                    return PARAM_INVALID);
  ge::GeTensorPtr filter = weights_pooling[0];
  FUSION_PASS_CHECK(filter == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filter is nullptr!"),
                    return PARAM_INVALID);
  int8_t* filterInt8Data = (int8_t*)(filter->GetData().data());
  vector<int64_t> filterDims = {inputC, 1, windowH, windowW};

  /* Store the filter data after optimization */
  std::unique_ptr<int32_t[]> weightInt8OutParam;
  Status ret = GetWeightOfConv(poolingNode->GetName(), filterInt8Data, filterDims, weightInt8OutParam);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get weight of conv failed."),
                    return ret);
  int64_t co = filterDims.at(INDEX_CO);

  // do not have bias, create bias node and init bias
  // And in this case, we do not need to shared the bias with other conv
  // So just create a new bias and set the data.

  OP_LOGD(FUSED_OP_TYPE.c_str(), "cube [%s] has no bias, create bias and set data", poolingNode->GetName().c_str());
  OP_LOGD(FUSED_OP_TYPE.c_str(), "the cube node have %ld in data Anchors", poolingNode->GetAllInDataAnchors().size());

  string constOpName = poolingNode->GetName() + "_bias";
  ge::OpDescPtr constOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(constOpDesc = std::make_shared<ge::OpDesc>(constOpName, CONSTANTOPTAB), return FAILED);
  ge::GeTensorDesc constOutDesc;
  FUSION_PASS_CHECK(constOpDesc->AddOutputDesc(constOutDesc) != ge::GRAPH_SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "AddOutputDesc failed!"), return FAILED);
  constOpDesc->SetType(CONSTANTOPTAB);
  ge::NodePtr constNode = graph.AddNode(constOpDesc);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "cube node %s, the const node %s", poolingNode->GetName().c_str(),
          constNode->GetName().c_str());
  FUSION_PASS_CHECK(constNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constNode is nullptr"),
                    return PARAM_INVALID);
  fusionNodes.push_back(constNode);

  // bias is the name of the third input of conv2d in IR conv2d.h
  ge::graphStatus res = poolingNode->AddLinkFrom(2, constNode);

  FUSION_PASS_CHECK(
      res != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: add const node failed.", poolingNode->GetName().c_str()),
      return res);

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

  weights_pooling.push_back(biasPtr);
  ge::OpDescUtils::SetWeights(poolingNode, weights_pooling);

  // update the bias outputDesc of biasOpDesc
  ge::GeTensorDesc inputDesc0 = poolingOp->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  int biasInputIndex = 2;
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
  if (biasOpDesc->UpdateOutputDesc(0, biasOutputDesc) != ge::GRAPH_SUCCESS) {
    biasPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update output desc of BiasNode[%s] not success.",
                                   biasOpDesc->GetName().c_str());
    return FAILED;
  }
  // update the bias inputDesc of the convOpDesc
  ge::GeTensorDesc biasDesc = poolingOp->GetInputDesc(biasInputIndex);
  biasDesc.SetShape(biasShape);
  biasDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasDesc.SetOriginShape(biasShape);
  biasDesc.SetOriginDataType(ge::DT_INT32);
  biasDesc.SetDataType(ge::DT_INT32);
  if (poolingOp->UpdateInputDesc(2, biasDesc) != ge::GRAPH_SUCCESS) {
    biasPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update bias input desc of ConvNode[%s] not success.",
            poolingNode->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool PoolingFusionPass::IsMeanValueAllEqual(vector<int64_t> input, vector<int64_t> window, vector<int64_t> stride,
                                            vector<int64_t> pad, int64_t ceil_mode) {
  // input feature map and pad dim size is 4;window and stride dim size is 2;
  FUSION_PASS_CHECK(input.size() != 4, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input is invalid."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(window.size() != 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "window is invalid."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(stride.size() != 2, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "stride is invalid."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(pad.size() != 4, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pad is invalid."),
                    return PARAM_INVALID);

  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
  // calculate out_size_h and out_size_w
  if (ceil_mode == 0) {
    out_size_h = static_cast<int>(ceil(static_cast<float>(input[2] + pad[0] + pad[1] - window[0]) / stride[0])) + 1;
    out_size_w = static_cast<int>(ceil(static_cast<float>(input[3] + pad[2] + pad[3] - window[1]) / stride[1])) + 1;
  } else if (ceil_mode == 1) {
    out_size_h = static_cast<int>(floor(static_cast<float>(input[2] + pad[0] + pad[1] - window[0]) / stride[0])) + 1;
    out_size_w = static_cast<int>(floor(static_cast<float>(input[3] + pad[2] + pad[3] - window[1]) / stride[1])) + 1;
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ceil_mode is invalid, please check!");
    return PARAM_INVALID;
  }

  // If we have padding, ensure that the last pooling starts strictly
  // inside the image (instead of at the padding); otherwise clip the last.
  if (pad[0] != 0 || pad[1] != 0) {
    if ((out_size_h - 1) * stride[0] >= input[2] + pad[0]) {
      --out_size_h;
    }
    if ((out_size_w - 1) * stride[1] >= input[3] + pad[2]) {
      --out_size_w;
    }

    if ((out_size_h - 1) * stride[0] >= input[2] + pad[0]) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "CHECK_LT((out_size_h - 1) * stride_h, in_size_h + pad_top)");
    }
    if ((out_size_w - 1) * stride[1] >= input[3] + pad[2]) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "CHECK_LT((out_size_w - 1) * stride_w, in_size_w + pad_left)");
    }
  }

  int64_t h_start = 0;
  int64_t w_start = 0;
  int64_t h_end = 0;
  int64_t w_end = 0;
  int64_t area = 0;
  for (int64_t steps_h = 0; steps_h < out_size_h; steps_h++) {
    for (int64_t steps_w = 0; steps_w < out_size_w; steps_w++) {
      h_start = steps_h * stride[0] - pad[0];
      w_start = steps_w * stride[1] - pad[2];
      h_end = min(h_start + window[0], input[2] + pad[0]);
      w_end = min(w_start + window[1], input[3] + pad[2]);
      area = max((h_end - h_start) * (w_end - w_start), static_cast<int64_t>(1));
      if (area != window[0] * window[1]) {
        return false;
      }
    }
  }

  return true;
}

vector<FusionPattern*> PoolingFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define PoolingFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PoolingMatrixFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_POOL, {POOLINGTAB}).SetOutput(PATTERN_POOL);

  patterns.push_back(pattern);
  return patterns;
}

Status PoolingFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get pooling node
  ge::NodePtr poolingNode = GetNodeFromMapping(PATTERN_POOL, mapping);
  FUSION_PASS_CHECK(poolingNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "poolingNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr poolingDesc = poolingNode->GetOpDesc();
  FUSION_PASS_CHECK(poolingDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "poolingNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // input of Pooling
  ge::GeTensorDesc poolingInputTensor = poolingNode->GetOpDesc()->GetInputDesc(0);

  // get Pooling input shape
  ge::GeShape poolingInputShape = poolingInputTensor.GetShape();
  // GESHAPE->vector
  vector<int64_t> dimInfo = poolingInputShape.GetDims();
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  if (dimInfo.size() == 4) {
    inputC = dimInfo[1];
    inputH = dimInfo[2];
    inputW = dimInfo[3];
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dimInfo is not right, please check!");
    return NOT_CHANGED;
  }

  if (PatternFusionUtil::IsUnknownShape(inputH) ||
      PatternFusionUtil::IsUnknownShape(inputC)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PoolingFusionPass cannot be applied for unknown shape.");
    return FAILED;
  }

  // get windowsize
  vector<int64_t> window;
  ge::AttrUtils::GetListInt(poolingDesc, "window", window);
  int64_t windowH = 0;
  int64_t windowW = 0;
  if (window.size() == 2) {
    windowH = window[0];
    windowW = window[1];
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "window is not right, please check!");
    return NOT_CHANGED;
  }
  // get stride
  vector<int64_t> stride;
  ge::AttrUtils::GetListInt(poolingDesc, "stride", stride);
  if (stride.size() != 2) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "stride is not right, please check!");
    return NOT_CHANGED;
  }
  // get pooling mode
  int64_t mode;
  ge::AttrUtils::GetInt(poolingDesc, "mode", mode);
  // get pooling ceil_mode
  int64_t ceil_mode;
  ge::AttrUtils::GetInt(poolingDesc, "ceil_mode", ceil_mode);
  if (ceil_mode != 0 && ceil_mode != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ceil_mode is not right, please check!");
    return NOT_CHANGED;
  }
  // get pooling attr global_pooling
  bool global_pooling;
  ge::AttrUtils::GetBool(poolingDesc, "global_pooling", global_pooling);
  // get pad
  vector<int64_t> pad;
  ge::AttrUtils::GetListInt(poolingDesc, "pad", pad);

  // check pad is zero or not
  bool isPadZero;
  Status retPad = IsPadZero(pad, isPadZero);
  FUSION_PASS_CHECK(retPad != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "IsPadZero failed."), return NOT_CHANGED);
  ge::GeTensorPtr assitPtr = nullptr;

  // judge pooling mode and global pooling
  string modeStr;
  if (mode == 0) {
    modeStr = "MAX";
  } else if (mode == 1) {
    modeStr = "AVG";
    if (global_pooling || ((inputH == windowH) && (inputW == windowW) && isPadZero)) {
      modeStr = "GAP";
    }
  }

  // only AVG pooling need generate assit matrix
  if (modeStr != "AVG") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "pooling mode is not AVG, graph not changed.");
    return NOT_CHANGED;
  }

  bool isWPadZero = false;
  if (pad.size() != 4) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the len of pad is not match.");
    return FAILED;
  }
  if (pad[2] == 0 && pad[3] == 0) {
    isWPadZero = true;
  }
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(poolingNode->GetOpDesc(), "groups", inputC),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Set groups attr failed."), return FAILED);
  // get pre node of pooling
  ge::InDataAnchorPtr poolingAnchorPtr0 = poolingNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr preAnchorPtr0 = poolingAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();

  bool IsInt8 = false;
  if (poolingNode->GetOpDesc()->GetInputDesc(0).GetDataType() == ge::DT_INT8) {
    IsInt8 = true;
  }

  if (inputH != windowH && inputW == windowW && isWPadZero) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputH != windowH && inputW == windowW , not support to call conv2d!");
    return NOT_CHANGED;
  }

  int64_t isaArchVer = 0;
  ge::AttrUtils::GetInt(poolingDesc, "isaArchVer", isaArchVer);
  if (isaArchVer == 1 && !IsMeanValueAllEqual(dimInfo, window, stride, pad, ceil_mode)) {
    if (preNode->GetOpDesc()->GetType() == "AscendQuant" ||
        preNode->GetOpDesc()->GetType() == "AscendAntiQuant" || IsInt8) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "in v200, area mean value is not a const, not support to call conv2d!");
      return NOT_CHANGED;
    }
  }

  if (preNode->GetOpDesc()->GetType() == "AscendQuant" || IsInt8) {
    // int8
    // if pooling's pre op is AscendQuant, pooling assitMatrix dtype is int8, and need add bias for pooling node
    ge::NodePtr dequantNode = nullptr;
    if (poolingNode->GetAllOutDataAnchors().empty()) {
      return PARAM_INVALID;
    }
    for (OutDataAnchorPtr outDataAnchor : poolingNode->GetAllOutDataAnchors()) {
      if (outDataAnchor == nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", poolingNode->GetName().c_str());
        return PARAM_INVALID;
      }
      if (outDataAnchor->GetPeerInDataAnchors().empty()) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr Out Data anchor.", poolingNode->GetName().c_str());
        return PARAM_INVALID;
      }
      for (InDataAnchorPtr inDataAnchorPtr : outDataAnchor->GetPeerInDataAnchors()) {
        if (inDataAnchorPtr == nullptr) {
          OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] has a nullptr in Data anchor.", poolingNode->GetName().c_str());
          return PARAM_INVALID;
        }
        dequantNode = inDataAnchorPtr->GetOwnerNode();
        if ((dequantNode->GetType() == "AscendDequant") || (dequantNode->GetType() == "AscendRequant")) {
          break;
        }
      }
    }
    if (dequantNode == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "poolingNode does not have a dequantNode output node.");
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

    int64_t matrixSize = inputC * windowH * windowW;
    FUSION_PASS_CHECK(matrixSize <= 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"),
                      return PARAM_INVALID);

    unique_ptr<int8_t[]> inputAssitInt8(new (std::nothrow) int8_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssitInt8.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "inputAssitInt8 is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(matrixSize, INT8_NUM_ONE, *reinterpret_cast<int8_t*>(inputAssitInt8.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                      return ret);
    vector<int64_t> assitDimInfoOrigin = {inputC, 1, windowH, windowW};
    if (!IsMeanValueAllEqual(dimInfo, window, stride, pad, ceil_mode)) {
      // judge for unknownshape
      ge::GeTensorDesc input_desc = dequantNode->GetOpDesc()->GetOutputDesc(0);
      ge::GeShape mulShape = input_desc.GetShape();
      vector<int64_t> dimMul = mulShape.GetDims();
      if (dimMul.size() >= 2 && PatternFusionUtil::IsUnknownShape(dimMul[1])) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PoolingFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      ge::NodePtr mulNode = AddMul(graph, dequantNode);
      FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      vector<int64_t> dimOut = inputDesc0.GetOriginShape().GetDims();
      if (dimOut.size() != 0) {
        for (size_t i = 1; i <= 3; i++) {
          auto dim = dimOut[i];
          if (PatternFusionUtil::IsUnknownShape(dim)) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "PoolingFusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
      }
      string recode;
      FUSION_PASS_CHECK(AddCoffe(graph, mulNode, pad, dimInfo, window, stride, recode, true) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return FAILED);
    }
    ge::GeTensorDesc tensorDesc;
    ge::GeShape assitShape(assitDimInfoOrigin);
    ge::GeShape assitShapeOrigin(assitDimInfoOrigin);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOrigin);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssitInt8.get()),
                                                   matrixSize * sizeof(int8_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
    FUSION_PASS_CHECK(!CheckOpSupported(poolingDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                      return NOT_CHANGED);
    ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assitPtr);
    FUSION_PASS_CHECK(const_opdesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to create const op desc."),
                      return FAILED);
    ge::NodePtr constNode = nullptr;
    constNode = graph.AddNode(const_opdesc);
    FUSION_PASS_CHECK(constNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add const node."),
                      return FAILED);
    FUSION_PASS_CHECK(poolingNode->AddLinkFrom(1, constNode) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Fail to link const node with pooling node."),
                      return FAILED);
    auto constInputNodes = OpDescUtils::GetConstInputs(poolingNode);
    NodePtr constInput = nullptr;
    if (constInputNodes.size() != 0) {
      constInput = constInputNodes[0];
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return PARAM_INVALID;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOPTAB);
    // add bias for pooling node
    ret = DoBiasOptimize(graph, poolingNode, fusionNodes, windowH, windowW, inputC);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "do fusion failed!"),
                      return ret);
  } else {  // fp16
    int64_t matrixSize = inputC * windowH * windowW;
    FUSION_PASS_CHECK(matrixSize <= 0, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize is Invalid"),
                      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[matrixSize]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(matrixSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                      return ret);

    vector<int64_t> assitDimInfoOrigin = {inputC, 1, windowH, windowW};
    float areaFactor = 0;

    if (IsMeanValueAllEqual(dimInfo, window, stride, pad, ceil_mode)) {
      areaFactor = 1.0 / (windowH * windowW);
      // generate one matrix
      ret = PoolingGenerateFilterFP16(matrixSize, areaFactor, *inputAssit.get());
    }
    else {
      areaFactor = 1.0;
      // generate one matrix
      ret = PoolingGenerateFilterFP16(matrixSize, areaFactor, *inputAssit.get());

      string pooling_name = poolingNode->GetName();
      string::size_type position = pooling_name.find("_mbatch_batch");
      string recode;
      if (position != string::npos) {
        recode = pooling_name.substr(0, position);
      }

      // judge for unknownshape
      ge::GeTensorDesc input_desc = poolingNode->GetOpDesc()->GetOutputDesc(0);
      ge::GeShape mulShape = input_desc.GetShape();
      vector<int64_t> dimMul = mulShape.GetDims();
      if (dimMul.size() >= 2 && PatternFusionUtil::IsUnknownShape(dimMul[1])) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PoolingFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      ge::NodePtr mulNode = AddMul(graph, poolingNode);
      FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "mulNode is null, AddMul failed."),
                        return PARAM_INVALID);
      ge::GeTensorDesc inputDesc0 = mulNode->GetOpDesc()->GetInputDesc(0);
      vector<int64_t> dimOut = inputDesc0.GetOriginShape().GetDims();
      if (dimOut.size() != 0) {
        for (size_t i = 1; i <= 3; i++) {
          auto dim = dimOut[i];
          if (PatternFusionUtil::IsUnknownShape(dim)) {
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                           "PoolingFusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
          }
        }
      }
      FUSION_PASS_CHECK(AddCoffe(graph, mulNode, pad, dimInfo, window, stride, recode, false) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return FAILED);
    }
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "GenerateFilterFP16 failed."),
                      return ret);

    ge::GeTensorDesc tensorDesc;
    ge::GeShape assitShape(assitDimInfoOrigin);
    ge::GeShape assitShapeOrigin(assitDimInfoOrigin);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc.SetOriginShape(assitShapeOrigin);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   matrixSize * sizeof(uint16_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
    FUSION_PASS_CHECK(!CheckOpSupported(poolingDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                      return NOT_CHANGED);
    ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assitPtr);
    FUSION_PASS_CHECK(const_opdesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to create const op desc."),
                      return FAILED);
    ge::NodePtr constNode = nullptr;
    constNode = graph.AddNode(const_opdesc);
    FUSION_PASS_CHECK(constNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add const node."),
                      return FAILED);
    FUSION_PASS_CHECK(poolingNode->AddLinkFrom(1, constNode) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Fail to link const node with pooling node."),
                      return FAILED);
    auto constInputNodes = OpDescUtils::GetConstInputs(poolingNode);
    NodePtr constInput = nullptr;
    if (constInputNodes.size() != 0) {
      constInput = constInputNodes[0];
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
      return PARAM_INVALID;
    }
    constInput->GetOpDesc()->SetType(CONSTANTOPTAB);
  }
  return SUCCESS;
}
REGISTER_PASS("PoolingFusionPass", BUILT_IN_GRAPH_PASS, PoolingFusionPass);
}  // namespace fe